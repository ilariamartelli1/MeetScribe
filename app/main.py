import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import ParamSpec, TypeVar

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.asr_contract import CPU_DEVICE, CUDA_DEVICE, Device
from app.infrastructure.config import Settings
from app.infrastructure.inference_gate import InferenceGate
from app.adapters.ollama_client import OllamaClient
from app.adapters.ollama_minutes import create_ollama_minutes_extractor
from app.adapters.model_factories import create_community1, create_faster_whisper
from app.core.meeting_minutes import (
    MeetingMinutesCoordinator,
    MinutesCoordinatorClosedError,
    MinutesEditValidationError,
    MinutesJobCapacityError,
    MinutesJobFinishedError,
    MinutesJobNotFoundError,
    MinutesJobNotReadyError,
    MinutesRegenerationConfirmationRequired,
    MinutesJobSnapshot,
    MinutesRevisionConflict,
    MinutesSourceUpdateConflict,
    render_minutes_text,
    render_transcript_text,
)
from app.core.minutes_contract import MinutesDraft
from app.core.transcription_jobs import (
    CoordinatorClosedError,
    JobError,
    JobBusyError,
    JobFinishedError,
    JobNotFoundError,
    JobState,
    ResultCapacityError,
    TranscriptionCoordinator,
)
from app.core.speaker_alignment import AttributedTranscript
from app.core.transcription_worker import SpeakerAttributedJob
from app.infrastructure.uploads import (
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
WEB_DIR = Path(__file__).resolve().parent / "web"
WEB_PATHS = {
    "/",
    "/transcript-download",
    "/static/app.js",
    "/static/transcript.js",
    "/static/styles.css",
}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'; object-src 'none'"
)


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
    configured_device = os.environ.get("MEETSCRIBE_DEVICE", CUDA_DEVICE)
    device: Device
    if configured_device == CPU_DEVICE:
        device = CPU_DEVICE
    elif configured_device == CUDA_DEVICE:
        device = CUDA_DEVICE
    else:
        raise ValueError("MEETSCRIBE_DEVICE must be 'cuda' or 'cpu'.")
    app.state.upload_dir = Path(tempfile.mkdtemp(prefix="meetscribe-uploads-"))
    inference_gate = InferenceGate()
    app.state.inference_gate = inference_gate
    try:
        app.state.coordinator = TranscriptionCoordinator(
            SpeakerAttributedJob(
                partial(
                    create_faster_whisper,
                    Path("models/faster-whisper-large-v3-turbo"),
                    device,
                ),
                partial(create_community1, Path("models/pyannote-community-1"), device),
            ),
            inference_gate=inference_gate,
        )
        app.state.ollama_client = OllamaClient(inference_gate)
        app.state.minutes_coordinator = MeetingMinutesCoordinator(
            create_ollama_minutes_extractor(app.state.ollama_client)
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
        try:
            await app.state.minutes_coordinator.close()
        finally:
            try:
                app.state.coordinator.shutdown()
                if app.state.stuck_validation is not None:
                    process, audio_path = app.state.stuck_validation
                    stop_validation_worker(process)
                    audio_path.unlink(missing_ok=True)
                if app.state.failed_upload_cleanup is not None:
                    app.state.failed_upload_cleanup.unlink(missing_ok=True)
                app.state.upload_dir.rmdir()
            finally:
                await app.state.ollama_client.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    error: JobError | None = None


class MinutesEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1, strict=True)
    minutes: dict[str, object]


class MinutesReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1, strict=True)


class MinutesRegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1, strict=True)
    confirm_overwrite: bool = Field(default=False, strict=True)


@app.middleware("http")
async def protect_job_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    jobs_path = f"{settings.api_prefix}/jobs"
    if request.url.path in WEB_PATHS:
        if request.url.hostname not in ("localhost", "127.0.0.1", "::1"):
            return Response(status_code=403, headers={"Cache-Control": "no-store"})
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    if request.url.path == jobs_path or request.url.path.startswith(f"{jobs_path}/"):
        origin = request.headers.get("origin")
        if (
            request.url.hostname not in ("localhost", "127.0.0.1", "::1")
            or (
                origin is not None
                and origin != f"{request.url.scheme}://{request.url.netloc}"
            )
            or (request.method in ("POST", "PUT", "PATCH", "DELETE") and origin is None)
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


@app.get("/", include_in_schema=False)
def browser_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/transcript-download", include_in_schema=False)
def transcript_download_page() -> FileResponse:
    return FileResponse(WEB_DIR / "transcript.html")


@app.get("/static/app.js", include_in_schema=False)
def browser_script() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")


@app.get("/static/transcript.js", include_in_schema=False)
def transcript_script() -> FileResponse:
    return FileResponse(WEB_DIR / "transcript.js", media_type="text/javascript")


@app.get("/static/styles.css", include_in_schema=False)
def browser_styles() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


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
            await request.app.state.minutes_coordinator.retain(
                active_coordinator.job_ids()
            )
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
def job_status(job_id: str, request: Request) -> JobStatusResponse:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        snapshot = coordinator.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobStatusResponse(
        job_id=snapshot.job_id,
        state=snapshot.state,
        error=snapshot.error,
    )


@app.post(f"{settings.api_prefix}/jobs/{{job_id}}/cancel")
def cancel_job(job_id: str, request: Request) -> JobStatusResponse:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        snapshot = coordinator.cancel(job_id)
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
    return JobStatusResponse(
        job_id=snapshot.job_id,
        state=snapshot.state,
        error=snapshot.error,
    )


def _get_attributed_transcript(
    coordinator: TranscriptionCoordinator, job_id: str
) -> AttributedTranscript:
    try:
        snapshot = coordinator.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if snapshot.state != "succeeded":
        raise HTTPException(
            status_code=409, detail="Minutes require a completed transcription job."
        )
    if not isinstance(snapshot.result, AttributedTranscript):
        raise HTTPException(
            status_code=409,
            detail="Speaker-aligned transcript is not available for this job.",
        )
    return snapshot.result


async def _retain_minutes_for_current_jobs(
    request: Request,
) -> MeetingMinutesCoordinator:
    minutes_coordinator: MeetingMinutesCoordinator = (
        request.app.state.minutes_coordinator
    )
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    await minutes_coordinator.retain(coordinator.job_ids())
    return minutes_coordinator


@app.post(f"{settings.api_prefix}/jobs/{{job_id}}/minutes", status_code=202)
async def start_minutes(job_id: str, request: Request) -> MinutesJobSnapshot:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    transcript = _get_attributed_transcript(coordinator, job_id)
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        return await minutes_coordinator.start(job_id, transcript)
    except MinutesJobCapacityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesSourceUpdateConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesCoordinatorClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    f"{settings.api_prefix}/jobs/{{job_id}}/minutes/regenerate",
    status_code=202,
    response_model=MinutesJobSnapshot,
)
async def regenerate_minutes(
    job_id: str, regeneration: MinutesRegenerateRequest, request: Request
) -> MinutesJobSnapshot:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        return await minutes_coordinator.regenerate(
            job_id,
            regeneration.expected_revision,
            confirm_overwrite=regeneration.confirm_overwrite,
        )
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        MinutesJobNotReadyError,
        MinutesRegenerationConfirmationRequired,
        MinutesRevisionConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesCoordinatorClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    f"{settings.api_prefix}/jobs/{{job_id}}/minutes",
    response_model=MinutesJobSnapshot,
)
async def minutes_status(job_id: str, request: Request) -> MinutesJobSnapshot:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        coordinator.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        return await minutes_coordinator.get(job_id)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    f"{settings.api_prefix}/jobs/{{job_id}}/minutes/cancel",
    response_model=MinutesJobSnapshot,
)
async def cancel_minutes(job_id: str, request: Request) -> MinutesJobSnapshot:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        return await minutes_coordinator.cancel(job_id)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinutesJobFinishedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesCoordinatorClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.patch(
    f"{settings.api_prefix}/jobs/{{job_id}}/minutes",
    response_model=MinutesJobSnapshot,
)
async def update_minutes(
    job_id: str, edit: MinutesEditRequest, request: Request
) -> MinutesJobSnapshot:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        minutes = MinutesDraft.model_validate_json(
            json.dumps(edit.minutes, ensure_ascii=False)
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Minutes do not match the required schema."
        ) from exc
    try:
        return await minutes_coordinator.update(job_id, edit.expected_revision, minutes)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinutesJobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesEditValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    f"{settings.api_prefix}/jobs/{{job_id}}/minutes/review",
    response_model=MinutesJobSnapshot,
)
async def review_minutes(
    job_id: str, review: MinutesReviewRequest, request: Request
) -> MinutesJobSnapshot:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        return await minutes_coordinator.mark_reviewed(job_id, review.expected_revision)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinutesJobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MinutesRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}/minutes.txt")
async def download_minutes_text(job_id: str, request: Request) -> Response:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        export = await minutes_coordinator.export(job_id)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinutesJobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        render_minutes_text(export),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="meetscribe-minutes.txt"'
        },
    )


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}/minutes.json")
async def download_minutes_json(job_id: str, request: Request) -> Response:
    minutes_coordinator = await _retain_minutes_for_current_jobs(request)
    try:
        export = await minutes_coordinator.export(job_id)
    except MinutesJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinutesJobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        export.model_dump_json(indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="meetscribe-minutes.json"'
        },
    )


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}/transcript.txt")
def download_transcript_text(job_id: str, request: Request) -> Response:
    coordinator: TranscriptionCoordinator = request.app.state.coordinator
    try:
        snapshot = coordinator.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if snapshot.state != "succeeded" or snapshot.result is None:
        raise HTTPException(
            status_code=409, detail="Transcript is not available for this job."
        )
    return Response(
        render_transcript_text(snapshot.result),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="meetscribe-transcript.txt"'
        },
    )
