# MeetScribe

MeetScribe is a local, Dockerized meeting assistant in development. It will first transcribe short WAV/MP3 recordings in a browser, then add speaker labels, editable draft minutes, and saved meetings. Audio and transcripts will be processed locally, using only synthetic or explicitly authorized non-sensitive test recordings during development.

**Current status:** Slice 0.1 is a minimal local FastAPI app with a health endpoint. It does not yet transcribe recordings or run in Docker; see [plan.md](plan.md) for the reviewable checkpoints.

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

Run pytest as `python -m pytest` so the repository root is on the import path. Model selection, download commands, CUDA/runtime versions, Docker commands, and data handling will be documented when those components are implemented; no model is used in this slice. No credentials, recordings, or weights belong in this repository.
