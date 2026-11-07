# MeetScribe

MeetScribe is a local, Dockerized meeting assistant in development. It will first transcribe short WAV/MP3 recordings in a browser, then add speaker labels, editable draft minutes, and saved meetings. Audio and transcripts will be processed locally, using only synthetic or explicitly authorized non-sensitive test recordings during development.

**Current status:** A minimal FastAPI app exposes a health endpoint and runs in Docker with either GPU access or an explicit CPU test configuration. No transcription is implemented yet; see [plan.md](plan.md) for the reviewable checkpoints.

## Local setup

Tested with Python 3.12.3 and [uv](https://docs.astral.sh/uv/) 0.11.6 on Linux. `uv.lock` pins the full dependency tree; direct requirements are FastAPI 0.115.12, Uvicorn 0.34.3, pytest 8.3.5, HTTPX 0.28.1, Ruff 0.11.13, and mypy 1.15.0. Install the exact locked dependencies:

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
MEETSCRIBE_DOCKER_TESTS=1 uv run --locked python -m pytest -q test/test_docker.py
```

Model selection, download commands, CUDA/runtime versions, and data handling will be documented when those components are implemented. No credentials, recordings, or weights belong in this repository.
