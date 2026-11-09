import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.transcription_jobs import (
    CoordinatorClosedError,
    JobBusyError,
    JobFinishedError,
    JobNotFoundError,
    JobSnapshot,
    ResultCapacityError,
    TranscriptionCoordinator,
)
from app.transcription_worker import FasterWhisperJob
from app.uploads import (
    MAX_REQUEST_BYTES,
    UploadError,
    UploadParser,
    ValidationWorkerStuck,
    stop_validation_worker,
    validate_audio,
)

settings = Settings()
logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


async def run_until_finished(
    function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    task = asyncio.create_task(run_in_threadpool(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


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
    app.state.upload_dir = Path(tempfile.mkdtemp(prefix="meetscribe-uploads-"))
    try:
        app.state.coordinator = TranscriptionCoordinator(
            FasterWhisperJob(Path("models/faster-whisper-large-v3-turbo"), device)
        )
    except BaseException:
        app.state.upload_dir.rmdir()
        raise
    # ponytail: one upload at a time; use bounded reservations for parallel uploads.
    app.state.upload_lock = asyncio.Lock()
    app.state.stuck_validation = None
    app.state.failed_upload_cleanup = None
    try:
        yield
    finally:
        app.state.coordinator.shutdown()
        if app.state.stuck_validation is not None:
            process, audio_path = app.state.stuck_validation
            stop_validation_worker(process)
            audio_path.unlink(missing_ok=True)
        if app.state.failed_upload_cleanup is not None:
            app.state.failed_upload_cleanup.unlink(missing_ok=True)
        app.state.upload_dir.rmdir()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def protect_job_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    jobs_path = f"{settings.api_prefix}/jobs"
    if request.url.path == jobs_path or request.url.path.startswith(f"{jobs_path}/"):
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


@app.post(f"{settings.api_prefix}/jobs", status_code=202)
async def upload_job(request: Request) -> JSONResponse:
    lengths = request.headers.getlist("content-length")
    if not lengths:
        raise HTTPException(status_code=411, detail="Content-Length is required.")
    if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
        raise HTTPException(status_code=400, detail="Invalid Content-Length.")
    if len(lengths[0]) > 12:
        raise HTTPException(status_code=413, detail="Request is too large.")
    length = int(lengths[0])
    if length > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Request is too large.")

    upload_lock: asyncio.Lock = request.app.state.upload_lock
    if (
        request.app.state.stuck_validation is not None
        or request.app.state.failed_upload_cleanup is not None
    ):
        raise HTTPException(
            status_code=503, detail="Upload processing is unavailable; restart the app."
        )
    if upload_lock.locked():
        raise HTTPException(status_code=409, detail="Another upload is in progress.")
    async with upload_lock:
        parser: UploadParser | None = None
        transferred = False
        coordinator: TranscriptionCoordinator | None = None
        try:
            parser = UploadParser(
                request.headers.get("content-type", ""), request.app.state.upload_dir
            )
            received = 0
            async with asyncio.timeout(120):
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > length or received > MAX_REQUEST_BYTES:
                        raise UploadError("Request is too large.", 413)
                    await run_until_finished(parser.write, chunk)
            if received != length:
                raise UploadError("Content-Length does not match the request.", 400)
            path, language, confirmation, audio_format = await run_until_finished(
                parser.finish
            )
            await run_until_finished(validate_audio, path, audio_format)
            active_coordinator: TranscriptionCoordinator = request.app.state.coordinator
            coordinator = active_coordinator
            job_id = await run_until_finished(
                active_coordinator.submit,
                path,
                language,
                replace_oldest_job_id=confirmation,
                cleanup_audio=True,
            )
            transferred = True
            return JSONResponse({"job_id": job_id}, status_code=202)
        except ValidationWorkerStuck as exc:
            assert parser is not None
            transferred = True
            request.app.state.stuck_validation = (exc.process, parser.path)
            logger.exception("Could not stop the audio validation worker.")
            raise HTTPException(
                status_code=503,
                detail="Audio validation is unavailable; restart the app.",
            ) from exc
        except UploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail="Upload timed out.") from exc
        except ResultCapacityError as exc:
            return JSONResponse(
                {"detail": str(exc), "oldest_job_id": exc.oldest_job_id},
                status_code=409,
            )
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CoordinatorClosedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            logger.exception("Could not process the uploaded audio.")
            raise HTTPException(
                status_code=503, detail="Could not process the uploaded audio."
            ) from exc
        finally:
            if (
                parser is not None
                and not transferred
                and (coordinator is None or not coordinator.manages_audio(parser.path))
            ):
                try:
                    await run_until_finished(parser.discard)
                except OSError as exc:
                    request.app.state.failed_upload_cleanup = parser.path
                    logger.exception("Could not remove temporary uploaded audio.")
                    raise HTTPException(
                        status_code=503,
                        detail="Temporary audio cleanup failed; restart the app.",
                    ) from exc


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
