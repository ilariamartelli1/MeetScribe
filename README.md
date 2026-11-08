# MeetScribe

MeetScribe is a local, Dockerized meeting assistant in development. It will first transcribe short WAV/MP3 recordings in a browser, then add speaker labels, editable draft minutes, and saved meetings. Audio and transcripts will be processed locally, using only synthetic or explicitly authorized non-sensitive test recordings during development.

**Current status:** A minimal FastAPI app exposes a health endpoint and runs in Docker with either GPU access or an explicit CPU test configuration. A typed transcription service and faster-whisper adapter are available for local use; there is no transcription API or browser workflow yet. See [plan.md](plan.md) for the reviewable checkpoints.

## Local setup

Tested with Python 3.12.3 and [uv](https://docs.astral.sh/uv/) 0.11.6 on Linux. `uv.lock` pins the full Python dependency tree. Install the exact locked dependencies:

```sh
uv sync --locked --group dev
```

Run the app locally in one terminal and check it in another:

```sh
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/api/v1/health
```

The health endpoint returns `{"status":"ok"}`. Run the tests and checks:

```sh
uv run --locked python -m pytest -q
uv run --locked ruff check app test
uv run --locked ruff format --check app test
uv run --locked mypy
```

Run pytest as `python -m pytest` so the repository root is on the import path.
Unit tests live in `test/unit/`; Docker and GPU integration tests live in
`test/integration/` and are opt-in. The CLion **Unit Tests** configuration can
target `test/unit`, and **Integration Test** can target `test/integration` with
`MEETSCRIBE_ASR_TESTS=1` and `MEETSCRIBE_DOCKER_TESTS=1`.

## Docker

The CPU test path was verified on Linux with Docker 29.6.1 and Docker Compose 5.3.0. It runs the health app without requesting a GPU; it does not demonstrate CPU transcription.

```sh
docker compose -f compose.yaml -f compose.cpu.yaml up --build -d
docker compose -f compose.yaml -f compose.cpu.yaml logs app
```

Wait until the logs show `Application startup complete`, then check the endpoint and stop the container:

```sh
curl --fail http://127.0.0.1:8000/api/v1/health
docker compose -f compose.yaml -f compose.cpu.yaml down
```

The default configuration requests NVIDIA GPUs and checks for GPU visibility before starting the app:

```sh
docker compose up --build -d
docker compose logs app
```

After the logs show `Application startup complete`, check GPU visibility and the endpoint, then stop the container:

```sh
docker compose exec app nvidia-smi -L
curl --fail http://127.0.0.1:8000/api/v1/health
docker compose down
```

GPU mode requires a working host NVIDIA driver and NVIDIA Container Toolkit configured for Docker. GPU visibility and the health endpoint were verified in the container on the development host with an NVIDIA GeForce RTX 4050 Laptop GPU. GPU visibility alone does not prove CUDA inference works.

Both configurations publish the app on `127.0.0.1:8000`, not a LAN interface. Set `MEETSCRIBE_PORT` in the shell before starting Compose to choose a different host port, and use that port in the health-check URL. To run the opt-in CPU container smoke test (which builds an image and creates and removes an isolated Compose project):

```sh
MEETSCRIBE_DOCKER_TESTS=1 uv run --locked python -m pytest -q test/integration/test_docker.py
```

## GPU transcription feasibility

The feasibility check uses faster-whisper 1.2.1, CTranslate2 4.8.5, PyAV 18.1.0, NVIDIA cuBLAS 12.9.3.13 and cuDNN 9.28.1.4. PyAV is pinned below 19 because faster-whisper 1.2.1 uses an `av.open` argument removed in PyAV 19. The CUDA libraries are installed in the app image, not required for the local CPU unit tests.

Provision the public [CTranslate2-converted large-v3-turbo model](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) separately. This downloads approximately 1.6 GB into the gitignored `models/` directory; it does not require a token:

```sh
HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  uv run --locked hf download deepdml/faster-whisper-large-v3-turbo-ct2 \
  config.json model.bin tokenizer.json preprocessor_config.json vocabulary.json \
  --revision 4df90f75321148c3a29a9e2351b7ddf8f5b115a8 \
  --local-dir models/faster-whisper-large-v3-turbo
```

The model card identifies the converted model as MIT-licensed. Its pinned `model.bin` SHA-256 is `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`; the check verifies this hash before loading. The faster-whisper and CTranslate2 projects are MIT-licensed, PyAV is BSD-licensed, and NVIDIA CUDA libraries have NVIDIA license terms. Review licenses before distribution.

The repository includes a short synthetic English sample in WAV and MP3 at `test/data/asr/`. It says: "This is a test meeting. The project is on schedule." It was regenerated locally with the Flite `cmu_us_kal` voice using the same text and method as the original temporary sample; the MP3 was encoded from the WAV using PyAV. The original files were deleted, so byte-for-byte identity with the original run cannot be verified.

After provisioning the model above, run the opt-in check:

```sh
MEETSCRIBE_ASR_TESTS=1 \
  uv run --locked python -m pytest -q -s test/integration/test_asr_gpu_feasibility.py
```

The test defaults to the provisioned model path and versioned fixtures. To use different local paths, set `MEETSCRIBE_MODEL_DIR`, `MEETSCRIBE_TEST_WAV`, or `MEETSCRIBE_TEST_MP3` to absolute paths. Use only synthetic or explicitly authorized non-sensitive audio; never commit meeting recordings. The container runs as a non-root user and must be able to read the mounted files.

ASR integration tests automatically build the app image once per pytest session, including when started from CLion. The build can require network access if dependencies are not cached. Inference runs with a disabled network and read-only mounts for the test tooling in `test/src/`, the model and the audio. The tooling is not included in the image. The feasibility test explicitly requests CUDA `int8_float16` and runs two jobs that each load the model, decode and transcribe both formats, then unload it. It reports timings and memory without printing transcript text. GPU memory is summed across visible GPUs and includes other processes; host usage is estimated from `/proc/meminfo`, while container usage comes from cgroup v2. Peaks are sampled approximately every 0.2 seconds, so brief spikes may be missed. The test requires an NVIDIA GPU and Linux cgroup v2, and does not establish transcription quality or long-recording capacity.

Run a separate offline correctness check against the versioned WAV and MP3:

```sh
MEETSCRIBE_ASR_TESTS=1 uv run --locked python -m pytest -q test/integration/test_asr_accuracy.py
```

This compares each transcript with the known sentence, ignoring capitalization, punctuation and whitespace. On a mismatch it reports the expected and actual text. It always uses the versioned synthetic audio, though `MEETSCRIBE_MODEL_DIR` can select a different local copy of the pinned model. Passing one synthetic sentence does not establish overall transcription quality.

## Timestamped transcription contract

`TranscriptionService` in `app/asr_service.py` accepts a `Transcriber` implementation and returns immutable transcript segments in integer milliseconds on the original recording timeline. It accepts English (`en`, default) or Italian (`it`), rejects silent/empty results, and validates segment ordering and bounds. Model failures propagate to the caller. `FasterWhisperTranscriber` in `app/faster_whisper_asr.py` implements the contract using the pinned local model, integrated VAD and transcription (not translation). It checks the weights SHA-256, requires an explicit `cuda` or `cpu` device and records model and inference settings with each result. CUDA does not silently fall back to CPU; call the adapter's `close()` when finished to unload the model.

The opt-in adapter smoke test runs the production adapter in the actual app image without network access:

```sh
MEETSCRIBE_ASR_TESTS=1 uv run --locked python -m pytest -q -s test/integration/test_asr_adapter.py
```

The versioned `quiet-negations-en.wav` and `quiet-negations-it.wav` fixtures in `test/data/asr/` are synthetic eSpeak NG 1.51 speech, with the negation utterances at 30% amplitude and long silent gaps. The English reference is "No. We do not approve." and the Italian reference is "No. Non siamo pronti." Annotated source intervals are `no` 3000–3321 ms and `not` 6647–6940 ms for English; `no` 3000–3265 ms and `non` 5965–6333 ms for Italian. The test checks each expected negation appears in a segment overlapping its source interval, and checks silence is rejected. Segment overlap does not establish word-level timing accuracy.

On November 8, 2026, the offline GPU smoke produced "No, we do not approve." and "No. Non siamo pronti." with no missed reference negations; silence was rejected. This small synthetic sample is not a quality benchmark, and quiet speech can still be missed in real recordings. Use only synthetic or explicitly authorized non-sensitive audio.

### Measured development-host result

On November 7, 2026, the offline check passed with a 3.531-second synthetic speech sample in both formats on Linux Mint 22.1, an RTX 4050 Laptop GPU (6,141 MiB), NVIDIA driver 580.159.03, NVIDIA Container Toolkit 1.20.1, Docker 29.6.1 and Compose 5.3.0. The image used Python 3.12.3, the dependency versions above, CUDA NVRTC 12.9.86, ONNX Runtime 1.31.1 and PyAV's bundled FFmpeg libraries (`libavcodec` 62.28.102, `libavformat` 62.12.102, `libavutil` 60.26.102). The pinned base image and uv image digests are in the Dockerfile; the tested local app image ID was `sha256:ac03ea7d3edaf1aa384443f91641829276f82c5a5589cc742438e41733d9d513`. Host drivers and toolkit are observed prerequisites, not managed by the image.

| Job | Model load | WAV decode / inference | MP3 decode / inference | Unload |
|---|---:|---:|---:|---:|
| 1 | 1.833 s | 0.020 / 1.125 s | 0.010 / 0.361 s | 0.094 s |
| 2 | 2.590 s | 0.009 / 0.740 s | 0.012 / 0.389 s | 0.076 s |

Both jobs reported effective device `cuda` and compute type `int8_float16`. Sampled peak usage was 1,344 MiB GPU, 11,158 MiB estimated host and 3,762 MiB container. Baselines were 55 MiB GPU, 9,072 MiB host and 1,577 MiB container; the container baseline includes the weight-file checksum read and its cache. After the first and second unloads, GPU usage was 150 MiB in each case, estimated host usage was 10,368 / 10,326 MiB, and container usage was 2,692 / 2,869 MiB. After the process exited, GPU usage was 52 MiB. These are whole-system/container observations, not isolated model allocations; retained page cache and other desktop activity affect them. Longer recordings and transcription accuracy still require separate evaluation.

No credentials, meeting recordings, or weights belong in this repository; the versioned audio above is synthetic test data.
