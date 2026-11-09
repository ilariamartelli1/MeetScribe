import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.config import Settings
from app.transcription_jobs import (
    CoordinatorClosedError,
    JobFinishedError,
    JobNotFoundError,
    JobSnapshot,
    TranscriptionCoordinator,
)
from app.transcription_worker import FasterWhisperJob

settings = Settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configured_device = os.environ.get("MEETSCRIBE_DEVICE", "cuda")
    device: Literal["cpu", "cuda"]
    if configured_device == "cpu":
        device = "cpu"
    elif configured_device == "cuda":
        device = "cuda"
    else:
        raise ValueError("MEETSCRIBE_DEVICE must be 'cuda' or 'cpu'.")
    app.state.coordinator = TranscriptionCoordinator(
        FasterWhisperJob(Path("models/faster-whisper-large-v3-turbo"), device)
    )
    try:
        yield
    finally:
        app.state.coordinator.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def protect_job_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.url.path.startswith(f"{settings.api_prefix}/jobs/"):
        origin = request.headers.get("origin")
        if (
            request.url.hostname not in ("localhost", "127.0.0.1", "::1")
            or (
                origin is not None
                and origin != f"{request.url.scheme}://{request.url.netloc}"
            )
            or (request.method == "POST" and origin is None)
        ):
            return JSONResponse(
                {"detail": "A same-origin localhost request is required."},
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    return await call_next(request)


@app.get(f"{settings.api_prefix}/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}")
def job_status(job_id: str, request: Request) -> JobSnapshot:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        return coordinator.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/jobs/{{job_id}}/cancel")
def cancel_job(job_id: str, request: Request) -> JobSnapshot:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        return coordinator.cancel(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobFinishedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CoordinatorClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        logger.exception("Could not stop the transcription worker.")
        raise HTTPException(
            status_code=503, detail="Could not stop the worker; retry cancellation."
        ) from exc
