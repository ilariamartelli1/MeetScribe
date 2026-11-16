# Sprint 2: Community-1 feasibility

## Status and scope

The current app image ran offline ASR followed by Community-1 diarization on
CUDA, and then separately on CPU without GPU access, twice each on a
3.531-second synthetic single-speaker WAV and MP3. Both ordinary and exclusive
speaker turns were returned. This verifies stack compatibility on these inputs,
not speaker-count or diarization accuracy. Speaker labels are not yet
integrated into the app.

The resource/headroom portion of slice 2.1 remains open: the machine already
used substantial swap, and system swap grew during the runs. An additional
nearly five-minute synthetic recording completed, but this does not establish
headroom under other desktop workloads. The test
decodes audio separately for TorchCodec validation, ASR and diarization; shared
decoding and its memory trade-offs remain to be addressed with the
diarization adapter.

## Provisioning and attribution

The gated [pyannote/speaker-diarization-community-1 model by pyannoteAI](https://huggingface.co/pyannote/speaker-diarization-community-1)
is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Review the model card and its access terms before use or redistribution. Its
requested research citations are Plaquet and Bredin, *Powerset multi-class
cross entropy loss for neural speaker diarization* (Interspeech 2023); Wang
et al., *Wespeaker: A research and production oriented speaker embedding
learning toolkit* (ICASSP 2023); and Landini et al., *Bayesian HMM clustering
of x-vector sequences (VBx) in speaker diarization: theory, implementation
and analysis on standard tasks* (Computer Speech & Language, 2022). The model
card has the full citations.

Provisioning uses a Hugging Face token after accepting the model conditions;
the test mounts no token or Hugging Face home. It uses `--network none`,
`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, and
`PYANNOTE_METRICS_ENABLED=0`. The pinned revision is
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`. The gitignored local directory
`models/pyannote-community-1` is mounted read-only at `/diarization`.
The following paths are relative to that directory, and the test checks their
SHA-256 hashes before loading:

| File | SHA-256 |
|---|---|
| `config.yaml` | `5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be` |
| `segmentation/pytorch_model.bin` | `7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e` |
| `embedding/pytorch_model.bin` | `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929` |
| `plda/plda.npz` | `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255` |
| `plda/xvec_transform.npz` | `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f` |

PyTorch's restricted checkpoint loader requires an explicit allowlist of three
pyannote task classes for these verified checkpoints. This does not make
arbitrary checkpoints safe to load.

## Historical CPU-diarization stack and limits

The image used pyannote.audio 4.0.0, PyTorch 2.8.0+cpu, torchaudio 2.8.0+cpu,
TorchCodec 0.7.0 and Debian FFmpeg `7:5.1.9-0+deb12u1`. FFmpeg is now pinned
to that package version in the Dockerfile; Debian repository availability and
unpinned system dependencies can still affect future builds. The existing ASR
stack used faster-whisper 1.2.1, CTranslate2 4.8.5, CUDA `int8_float16`,
PyAV 18.1.0, cuBLAS 12.9.3.13 and cuDNN 9.28.1.4.

After adding these dependencies, the existing four-recording ASR checkpoint
still passed its 3% WER and selected-negation gates: English WAV 1.72%,
English MP3 1.44%, Italian WAV 0.68% and Italian MP3 0.00%. This is a
synthetic ASR check, not an assessment of diarization quality.

The Linux development host has an NVIDIA RTX 4050 Laptop GPU with about 6 GiB
VRAM and about 15 GiB RAM. Docker limited the test container to 6 GiB memory
and four CPUs; PyTorch used two intra-op and two inter-op threads. ASR ran on
CUDA and PyTorch was CPU-only. The runner released model and waveform
references between stages. The test's Docker limits are not yet application
deployment limits.

| Job | ASR, seconds | Diarization, seconds | Regular / exclusive turns |
|---|---:|---:|---:|
| WAV 1 | 4.48 | 2.37 | 1 / 1 |
| MP3 1 | 3.29 | 1.13 | 1 / 1 |
| WAV 2 | 3.12 | 1.12 | 1 / 1 |
| MP3 2 | 3.11 | 1.13 | 1 / 1 |

Sampled memory in MiB, from one successful short-fixture run:

| Observation | GPU | Host used estimate | System swap used | Container |
|---|---:|---:|---:|---:|
| Baseline | 58 | 10,839 | 4,020 | 541 |
| Peak across ASR stages | 1,286 | 12,177 | 5,110 | 4,575 |
| Peak across diarization stages | 156 | 11,808 | 5,108 | 3,403 |
| After WAV 1 | 156 | 11,721 | 4,528 | 3,300 |
| After MP3 1 | 156 | 11,216 | 5,046 | 3,398 |
| After WAV 2 | 156 | 11,092 | 5,071 | 3,403 |
| After MP3 2 | 156 | 11,148 | 5,107 | 3,403 |

Each column's peak may occur at a different instant. GPU, host and swap
figures are whole-system observations, not allocations attributable solely to
the test; container cgroup memory includes cache. Sampling approximately every
0.2 seconds can miss brief peaks, including short codec stages. From baseline
to the observed maximum, system swap increased by about 1.1 GiB, but other
desktop activity and pre-existing memory pressure prevent assigning that
increase to the container alone. A successful capped short-input run does not
establish adequate headroom for longer recordings or other desktop workloads.

## Longer synthetic recording

The same runner was also used with the English checkpoint WAV (297.30 seconds)
and MP3 (297.38 seconds). This separate run used the same two-thread and
four-CPU limits but set `--memory 6g --memory-swap 6g`, disallowing container
swap. The recording is synthetic and single-speaker; returned turn counts do
not demonstrate correct speaker labels.

| Job | ASR, seconds | Diarization, seconds | Regular / exclusive turns |
|---|---:|---:|---:|
| WAV 1 | 9.98 | 292.30 | 12 / 4 |
| MP3 1 | 9.57 | 285.85 | 12 / 4 |
| WAV 2 | 9.71 | 280.15 | 12 / 4 |
| MP3 2 | 9.50 | 284.28 | 12 / 4 |

| Observation | GPU MiB | Host used estimate MiB | System swap used MiB | Container MiB |
|---|---:|---:|---:|---:|
| Baseline | 54 | 9,200 | 5,556 | 407 |
| Peak across ASR stages | 1,346 | 11,401 | 5,842 | 4,226 |
| Peak across diarization stages | 152 | 12,863 | 6,225 | 4,981 |
| After WAV 1 | 152 | 11,117 | 5,825 | 2,196 |
| After MP3 1 | 152 | 11,011 | 5,837 | 2,265 |
| After WAV 2 | 152 | 11,105 | 5,842 | 3,029 |
| After MP3 2 | 152 | 11,126 | 5,679 | 2,816 |

Diarization alone took nearly the recording's duration on this host. The
container completed within its no-swap cap, but system swap still increased
from an already high baseline; a no-swap container can still create memory
pressure on other processes. The same attribution and sampling limitations
apply as above. The pipeline also emitted PyTorch warnings about standard
deviation with insufficient degrees of freedom; the test checked nonempty
outputs, not whether those warnings affected accuracy.

These are historical CPU-diarization measurements from commit `4224dfd`.
The current runner selects the same device for ASR and diarization, so its CPU
mode does not reproduce this mixed GPU-ASR/CPU-diarization run. See the README
for provisioning and the current opt-in test commands.

## CUDA-enabled stack: short-input check

The current image uses PyTorch and torchaudio 2.8.0+cu128, pyannote.audio
4.0.0, TorchCodec 0.7.0, FFmpeg `7:5.1.9-0+deb12u1`, cuBLAS 12.8.4.1 and
cuDNN 9.10.2.21. The NVIDIA libraries differ from the historical run above.
The runner selects CUDA or CPU for both stages, unloads ASR before diarization,
and checks the diarization pipeline, segmentation model and embedding device.
CUDA-unavailable mode fails explicitly; CPU mode ran in a container without
GPU access. On CPU, CTranslate2 resolves `int8` to `int8_float32` on this
host, so the app now requests and verifies `int8_float32` explicitly.

Both formats were processed twice in each mode with a 6 GiB container limit,
four CPUs and two PyTorch threads. Timings include model loading, decoding,
inference and cleanup; they are not isolated inference benchmarks.

| Job | CUDA ASR / diarization (seconds) | CPU ASR / diarization (seconds) |
|---|---:|---:|
| WAV 1 | 3.83 / 1.82 | 10.22 / 2.27 |
| MP3 1 | 3.37 / 0.46 | 10.53 / 1.19 |
| WAV 2 | 3.77 / 0.43 | 10.54 / 1.18 |
| MP3 2 | 3.30 / 0.44 | 9.97 / 1.20 |

| CUDA run observation | GPU MiB | Host used estimate MiB | System swap used MiB | Container MiB |
|---|---:|---:|---:|---:|
| Baseline | 61 | 8,285 | 3,969 | 480 |
| Peak across ASR stages | 1,339 | 11,388 | 3,971 | 3,956 |
| Peak across diarization stages | 343 | 10,301 | 3,971 | 2,853 |
| After final diarization | 211 | 10,265 | 3,971 | 2,852 |

Each column's peak may occur at a different instant, and system-wide and
sampling limitations described above apply. The test did not disable container
swap. The synthetic outputs are not a diarization accuracy assessment.

After the dependency change, the four-recording GPU ASR checkpoint passed:
English WAV 1.72%, English MP3 1.44%, Italian WAV 0.68% and Italian MP3
0.00% WER, with no selected-negation differences. This does not establish
accuracy on real meetings.

## CPU/GPU comparison: nearly five-minute MP3

The versioned `test/data/checkpoint/english/meeting-en.mp3` is a synthetic,
single-speaker recording lasting 297.38 seconds (4:57). Its SHA-256 is
`ef3d4e68a1be6e18fc08788d08e363e3b75f6246a510ae488f10543dde4db69c`.
It was run once per device on the same host and current CUDA-enabled image,
with CPU and GPU runs sequential rather than concurrent. Both stages used
the selected device; CPU mode had no GPU exposed. The host was an RTX 4050
Laptop GPU (6,141 MiB VRAM) with approximately 15 GiB system RAM. Each
container had four CPUs, two PyTorch threads and a 6 GiB memory limit with
container swap disabled. ASR used `int8_float32` on CPU and `int8_float16`
on CUDA. These are one-run observations, not statistically stable benchmarks.

| Stage | CPU | GPU | CPU / GPU time |
|---|---:|---:|---:|
| Transcription | 89.07 s | 9.74 s | 9.1× |
| Diarization | 264.20 s | 16.30 s | 16.2× |
| Sum of stages | 353.27 s | 26.04 s | 13.6× |

These are elapsed stage times, including model loading, decoding, inference
and cleanup; the sum is not total wall-clock time, because it excludes process
startup and the separate TorchCodec check. CUDA is synchronized before timing
diarization completion. Both runs returned 12 ordinary and four exclusive
turns, but counts alone do not measure whether speakers were identified
correctly. The CPU and GPU runs both emitted PyTorch warnings about standard
deviation with insufficient degrees of freedom; accuracy was not evaluated.

| Observation | CPU host MiB | CPU swap MiB | CPU container MiB | GPU host MiB | GPU swap MiB | GPU container MiB | GPU VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 8,519 | 3,728 | 463 | 8,714 | 3,733 | 479 | 61 |
| ASR stage peak | 11,335 | 3,728 | 2,906 | 10,709 | 3,733 | 2,219 | 1,353 |
| Diarization stage peak | 11,680 | 3,720 | 3,285 | 10,704 | 3,733 | 2,255 | 2,201 |
| After diarization | 10,200 | 3,713 | 2,039 | 10,680 | 3,732 | 2,255 | 211 |

The host and swap figures cover the whole computer, and the container figures
include cache. Each column's peak may occur at a different instant. Sampling
every approximately 0.2 seconds can miss peaks; CPU mode did not measure GPU
memory. Neither run used container swap, but system swap was already occupied
by other activity. These observations do not establish capacity or accuracy
for 30-minute meetings or prove responsiveness under every desktop workload.

After provisioning the models as described in the README, the GPU run can be
repeated with:

```sh
docker compose build app
docker run --rm --network none --gpus all --memory 6g --memory-swap 6g --cpus 4 \
  --mount "type=bind,src=$PWD/models/faster-whisper-large-v3-turbo,dst=/model,readonly" \
  --mount "type=bind,src=$PWD/models/pyannote-community-1,dst=/diarization,readonly" \
  --mount "type=bind,src=$PWD/test/data/checkpoint/english/meeting-en.mp3,dst=/audio/synthetic-meeting.mp3,readonly" \
  --mount "type=bind,src=$PWD/test/src,dst=/checks,readonly" \
  --env PYTHONPATH=/app:/checks --entrypoint python meetscribe-app \
  /checks/diarization_feasibility.py --device cuda --format mp3 --runs-per-format 1
```

For CPU mode, omit `--gpus all` and replace `--device cuda` with `--device cpu`.
