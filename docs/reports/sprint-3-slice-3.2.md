# Sprint 3.2 — GPU handoff

## Shared gate, in brief

One FastAPI-owned gate serializes GPU work. `TranscriptionCoordinator` holds it
from before worker start through confirmed exit and cleanup. `OllamaClient`
waits for that same gate, then holds it through generation and verified model
unload. Only then can next inference start. This is not a job queue or
parallel scheduler; it prevents ASR/diarization and Ollama from competing for
GPU memory. If unload cannot be verified, the gate blocks later work.

`InferenceGate` is MeetScribe's app-specific mutex wrapper around Python's
`threading.Lock`, with a state lock and an unavailable reason.
`try_acquire()` is non-blocking: transcription returns busy if occupied, while
Ollama retries asynchronously. It provides no first-in/first-out queue or
fairness guarantee.

The ASR worker is a spawned Python child inside the app container, not an
Ollama worker or separate container. It loads Whisper, transcribes, closes
Whisper, then loads and closes Community-1 for diarization. It sends its result
to the parent and exits; the coordinator confirms exit and cleanup before
releasing the gate. Ollama runs in a separate container: it loads Qwen from the
local model store for generation, then `keep_alive=0` requests unload. The
client waits until `/api/ps` confirms Qwen is gone before releasing the gate.
This trades model reload time for non-overlapping GPU use.

## Outcome

Added a shared inference gate for transcription and Ollama requests, with model
release verification before the next GPU job can start. The real-container
ASR -> diarization -> Ollama -> ASR test passed on the development RTX 4050.
Concurrent generations serialized, the cancelled generation released Ollama's
model, and all stages completed without OOM or CPU offload. This is one
synthetic-fixture run, not a capacity or accuracy guarantee. No performance
baseline was measured, so the run does not establish whether there was a
slowdown.

## Handoff design

- The FastAPI process owns one thread-safe gate shared by
  `TranscriptionCoordinator` and `OllamaClient`. The spawned transcription
  worker never owns the lock: the coordinator holds it from before process
  start until the monitor confirms the child has exited and cleanup is done.
- A transcription submission fails busy while Ollama holds the gate; Ollama
  generation waits asynchronously while transcription owns it.
- Before generation, `OllamaClient` verifies the installed model's exact
  manifest digest. It streams output with `keep_alive=0` and `think=false`,
  then polls `/api/ps` until the model is absent. Cancellation closes the
  request and shields release verification from repeated cancellation. If
  release cannot be verified, the gate is marked unavailable instead of
  allowing another GPU job into an uncertain state. A second cancellation
  during the unload check must not release the gate early: that could let ASR
  load while Qwen still occupies VRAM. The current client keeps checking; if
  unload remains unconfirmed, it blocks later GPU work.
- Compose sets Ollama to one loaded model and one parallel request. The
  integration test uses an isolated offline Compose project; it does not
  publish Ollama or expose the app port.

## Observed GPU handoff

This test is opt-in: it builds and runs real Docker containers, requires
Docker's NVIDIA Container Toolkit and provisioned ASR, Community-1 and Qwen3
models, and consumes host/GPU resources. A fresh app-image build took 24 min
31 s; the handoff run took 86.73 s. Default tests skip it unless enabled.

### Measurement method

These test-only measurements track GPU use across ASR, diarization and Ollama
handoffs, check memory after model release, and reveal host/container RAM or
swap pressure. They help assess resource behavior on this machine; they do
not establish transcription accuracy or capacity for longer meetings or a
future VM.

**First run — CLI commands and direct reads.** CLI commands used were
`nvidia-smi` for GPU memory and repeated Docker CLI calls for app/Ollama
container RAM. The archived run record does not retain the exact Docker
subcommands or arguments, so those cannot be listed reliably. Host RAM and
swap came from direct `/proc/meminfo` reads, not a CLI command. GPU readings
include the desktop baseline, not per-process usage. This was the first
real-container run of the same ASR -> diarization -> Ollama -> ASR sequence,
before the sampler was improved.

Its target sampling interval was 0.5 seconds, but it collected only 21
resource samples; observed gaps within active stages reached about 4.5–4.7
seconds. The run did not time each CLI call separately, so the exact share of
that gap caused by Docker CLI overhead is unknown. This sampling was too coarse
to reliably capture peaks during the 4–5-second ASR stages.

At that run's baseline, system-wide swap use was 3,449.75 MiB (3.37 GiB);
peak use was 6,628.5 MiB (6.47 GiB) out of 8.0 GiB total. The earlier label
“6.63 GiB” was a unit-conversion error. About 4.63 GiB was free at baseline
and 1.53 GiB remained free at the observed peak. This is total host swap in
use, not an additional 6.47 GiB consumed by the test or swap attributable
solely to it. The run completed without OOM, but no performance baseline was
measured, so it cannot establish whether there was a slowdown. Coarse sampling
could also miss brief peaks; these results are preliminary.

**Changes after the first run.** The corrected sampler's one-time CLI commands
are `docker compose ps -q <service>` and
`docker inspect --format '{{.State.Pid}}' <container-id>`; they locate each
container's cgroup v2 path. It then reads `memory.current` directly instead of
repeating Docker CLI calls for RAM. GPU and host readings still use
`nvidia-smi` and `/proc/meminfo`. A separate two-second placement sampler runs
the CLI command `docker exec <ollama-container-id> ollama ps` during Ollama
stages to record whether Qwen is on the GPU.

**Snapshot function and CLI command.** After the preliminary run, the test
added [`memory_snapshot()`](../../test/src/gpu_handoff_smoke.py#L50) for
point-in-time readings. Its only CLI command is
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`; it does not
call Docker CLI. It reads `/proc/meminfo` and
`/sys/fs/cgroup/memory.current` directly. ASR and diarization snapshots are
taken after inference returns but before the adapter closes; the Ollama
snapshot is taken on its first streamed output; post-stage snapshots show
memory after model release. These snapshots show model-resident memory, not
inference peaks. Periodic samples estimate peaks: the corrected run collected
156 samples, with no more than about 0.53 seconds between samples in active
stages. This gives multiple samples during 4–5-second ASR stages, though
spikes shorter than 0.53 seconds can still be missed.

**Corrected run.** Command:

```sh
MEETSCRIBE_GPU_HANDOFF_TESTS=1 PYTHONPATH=. \
  uv run --locked python -m pytest -q -s test/integration/test_gpu_handoff.py
```

Result: **1 passed in 86.73 seconds** using the versioned synthetic two-voice
recording and pinned ASR, Community-1 and Qwen3 models. ASR and diarization
reported CUDA; `ollama ps` showed Qwen3 at **100% GPU**, 3.2 GB, with a
4,096-token context. Ollama's loaded-model API was empty after each generation
and after cancellation.

The device reported 6,141 MiB total GPU memory. The host reported 15.3 GiB
physical RAM and 8.0 GiB total swap; sampled host use peaked at 11.1 GiB. At
the corrected run's baseline, 4.28 GiB of swap was already used, leaving
about 3.72 GiB free. System-wide swap use peaked at 5.12 GiB, leaving about
2.88 GiB free—an increase of about 0.84 GiB from baseline. The run completed
without OOM. This is a meaningful resource warning, not proven headroom on a
clean host or the future VM.

Corrected-run stage comparison:

| Stage | Elapsed | GPU used while active | GPU after stage | Sampled host RAM peak | Sampled swap peak |
|---|---:|---:|---:|---:|---:|
| ASR before Ollama | 5.449 s | 1,252 MiB snapshot; 1,284 MiB sampled | 186 MiB | 10,563 MiB | 4,421 MiB |
| Community-1 diarization and alignment | 10.280 s | 2,190 MiB | 288 MiB | 10,438 MiB | 4,420 MiB |
| Initial Ollama generation | 47.816 s | 3,439 MiB | 300 MiB | 11,099 MiB | 4,703 MiB |
| Two concurrent requests, serialized | 2.494 s + 4.381 s | 3,439 MiB for each request | 300 MiB | 10,740 MiB | 4,701 MiB |
| Cancelled Ollama generation | 1.987 s | 3,439 MiB snapshot; 2,763 MiB sampled | 314 MiB | 10,568 MiB | 4,701 MiB |
| ASR after Ollama | 4.273 s | 1,364 MiB snapshot; 1,428 MiB sampled | 300 MiB | 11,360 MiB | 5,244 MiB |

The test calls production ASR and diarization adapters directly inside a
measurement process in the app container to collect stage snapshots. Unit
tests separately verify that the real coordinator holds the same gate for its
spawned worker. This is not a browser-upload benchmark; the long-lived
measurement process also retains Python libraries that the production
transcription child releases on exit. The short synthetic recording does not
establish accuracy or two-hour throughput. The ASR-before/after timings are
single-run observations and do not establish a performance trend.

## Validation

| Check | Result |
|---|---|
| Focused gate, Ollama-client and coordinator tests | 37 passed |
| Default suite, `uv run --locked python -m pytest -q -r a` | 205 passed, 30 opt-in cases skipped, one existing Starlette/AnyIO deprecation warning |
| Corrected real-container GPU handoff test | 1 passed in 86.73 s |
| Fresh app image build | Passed; CUDA and diarization dependencies took 24 min 31 s to prepare, with 239 s exporting layers |
| Ruff, format, mypy, lockfile and Compose checks | Passed |

The default suite does not run the remaining opt-in integration checks. The
historical exact-text ASR failure and repeated-sentence XFAIL were not rerun or
changed; see [test status](../status/test-status.md).

## Next

Slice 3.3 defines typed minutes, citation and completeness checks independently
of the summary model.
