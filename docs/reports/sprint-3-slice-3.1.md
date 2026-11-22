# Sprint 3.1 — Private local summary model

## Outcome

Added an opt-in, digest-pinned Ollama service and provisioned the official
Qwen3 4B Q4_K_M model. A short app-container request completed on the NVIDIA
GPU with external networking blocked. This slice adds model infrastructure
only; the browser and job API do not generate summaries yet.

## Runtime design

- Ollama is started only with the Compose `summary` profile. The default app
  and CPU smoke paths do not start a GPU model service.
- The app and Ollama share the `inference` network, which is marked internal.
  Ollama has no published host port and no internet route in normal runtime;
  `OLLAMA_NO_CLOUD=1` also disables Ollama cloud features.
- The app retains the normal Compose network needed for the loopback-published
  browser port. The port mapping
  `127.0.0.1:${MEETSCRIBE_PORT:-8000}:8000` forwards the host's `127.0.0.1`
  port (8000 by default) to port 8000 inside the app container. Binding the
  host side to `127.0.0.1` lets a browser on the same computer reach MeetScribe
  through that mapping, but does not publish the port to other devices on the
  LAN. The opt-in offline test removes that network and the app port, leaving
  only the internal app-to-Ollama network.
- Model provisioning temporarily attaches Ollama to the normal Compose network
  so it can download the pinned library artifact. The model is stored in the
  gitignored `ollama/` directory; the provisioning container and networks are
  removed afterward.

An initial attempt to mark the whole default network internal blocked the
existing host-published app port. The runtime network was split as above, and
the existing Docker smoke test passed after the change.

## Pins and compatibility

| Stage | Runtime and artifact |
|---|---|
| ASR | faster-whisper 1.2.1 / CTranslate2 4.8.5; `deepdml/faster-whisper-large-v3-turbo-ct2`, revision `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` |
| Diarization | pyannote-audio 4.0.0 / PyTorch 2.8.0; Community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |
| Summary | Ollama 0.40.17, image index digest `sha256:ccc42bcd3d5fcf7ab0d532b300382154e1a4da56fc666bdb313b78ebdd912754`; tested Linux/amd64 image digest `sha256:5e5802d3c018ecdee7c7dff32a7e3b10ca9a2c4b3a461392cff97c313c2af0fc` |
| Summary model | `qwen3:4b`, Q4_K_M, 4.0B parameters, registry manifest digest `sha256:359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` |

Qwen3 4B is listed under Apache-2.0 in the official
[model card](https://huggingface.co/Qwen/Qwen3-4B). The pinned model layer is
2,497,280,480 bytes; the API reported 2,497,293,931 bytes across the complete
model artifact. The Ollama model reports an advertised context length of
262,144 tokens, but that is not the configured or measured application
budget. The smoke used a 4,096-token context.

The versioned pins and smoke settings are in
[`app/model_manifests/summary_model.toml`](../../app/model_manifests/summary_model.toml).
The compatibility rows record dependency and artifact pins; they do not prove
that the full ASR/diarization/summary pipeline fits concurrently.

## Smoke settings and measurements

The isolated offline generation used `num_ctx=4096`, `num_predict=16`,
`temperature=0`, JSON format, `think=false`, and `keep_alive=5m`. The model's
`/api/show` response reports thinking enabled by default; the request-level
`think=false` control was accepted and the response contained no thinking
text.

Setting notes:

- `num_ctx=4096`: prompt-plus-output context cap; tested smoke budget, not
  Qwen3's advertised 262,144-token maximum.
- `num_predict=16`: output cap; enough for tiny smoke reply ({"ready": true} -> 9 tokens). Minutes
  extraction later allows up to 768.
- `temperature=0`: minimize sampling randomness; easier smoke verification.
- JSON format: request JSON output; smoke parses exact object. Later extraction
  supplies full minutes schema.
- `think=false`: disable Qwen3's default thinking text and extra tokens.
- `keep_alive=5m`: retain model for follow-up `ollama ps` GPU-placement check.
  Slice 3.2 uses `0` to unload model before next GPU stage.

| Measurement | Observed |
|---|---:|
| Prompt tokens | 23 |
| Generated tokens | 9 |
| Completion reason | `stop` |
| Model load | 26,714 ms |
| Prompt evaluation | 19,528 ms |
| Generation | 1,688 ms |
| Ollama placement | 100% GPU; 37/37 layers offloaded |
| GPU memory snapshot | 3,246 MiB used, 2,526 MiB free |
| CUDA KV cache | 576 MiB on CUDA0 |
| Ollama container memory snapshot | 2.064 GiB of 15.33 GiB |
| External network | Blocked; internal inference network contained only app and Ollama |

These are single-run observations on the development RTX 4050 Laptop GPU
(6,141 MiB reported total, driver 580.159.03), not averages or throughput
benchmarks. Memory figures are snapshots, not sampled peaks. The run did not
include ASR or diarization, so it does not establish combined-pipeline
headroom, long-context capacity, VM requirements, or summary quality.

## Validation

| Check | Result |
|---|---|
| Default suite, `uv run --locked python -m pytest -q -r a` | 191 passed, 29 opt-in cases skipped, one existing Starlette/AnyIO deprecation warning |
| CPU Docker smoke, `MEETSCRIBE_DOCKER_TESTS=1 ... test/integration/test_docker.py` | 1 passed in 3.35 s; host loopback binding works |
| Offline real-model smoke, `MEETSCRIBE_OLLAMA_TESTS=1 ... test/integration/test_ollama_model.py` | 1 passed in 54.09 s; exact digest, Q4_K_M, internal-only connectivity, GPU placement and non-truncated JSON completion verified |
| Ruff, format check, mypy | Passed |
| Compose base, provisioning and offline configurations | Valid |

During test development, a free-text prompt reached the configured token limit
(`done_reason=length`). The final smoke uses constrained JSON, verifies the
response shape, and requires `done_reason=stop`; it is a runtime smoke, not a
meeting-summary quality test. The complete historical all-opt-in suite was
not rerun.

## Next

Slice 3.2 must serialize ASR, diarization, summarization and regeneration across
the shared GPU, unload the model before handing off to another stage, and
verify cancellation before admitting subsequent work.
