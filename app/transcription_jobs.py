import math
import multiprocessing
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal

from app.asr_contract import Language, Transcript

JobState = Literal["running", "cancelling", "cancelled", "succeeded", "failed"]
TranscriptionJob = Callable[[Path, Language], Transcript]


@dataclass(frozen=True)
class JobError:
    kind: str
    message: str


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    state: JobState
    result: Transcript | None = None
    error: JobError | None = None


@dataclass(frozen=True)
class _Outcome:
    result: Transcript | None = None
    error: JobError | None = None


@dataclass
class _Record:
    snapshot: JobSnapshot
    finished_at: float | None = None


class JobBusyError(RuntimeError):
    pass


class ResultCapacityError(RuntimeError):
    def __init__(self, oldest_job_id: str) -> None:
        self.oldest_job_id = oldest_job_id
        super().__init__(
            "Result capacity reached; confirm replacing the oldest result."
        )


class JobNotFoundError(LookupError):
    pass


class JobFinishedError(RuntimeError):
    pass


class CoordinatorClosedError(RuntimeError):
    pass


def _run_job(
    job: TranscriptionJob, audio_path: Path, language: Language, sender: Connection
) -> None:
    try:
        try:
            outcome = _Outcome(result=job(audio_path, language))
        except Exception as exc:
            outcome = _Outcome(error=JobError(type(exc).__name__, str(exc)))
        sender.send(outcome)
    finally:
        sender.close()


class TranscriptionCoordinator:
    def __init__(
        self,
        job: TranscriptionJob,
        *,
        result_lifetime_seconds: float = 3600,
        max_results: int = 16,
        shutdown_timeout_seconds: float = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not math.isfinite(result_lifetime_seconds)
            or result_lifetime_seconds <= 0
            or max_results < 1
        ):
            raise ValueError("Result lifetime and capacity must be positive.")
        if not math.isfinite(shutdown_timeout_seconds) or shutdown_timeout_seconds <= 0:
            raise ValueError("Shutdown timeout must be positive.")
        self._job = job
        self._lifetime = result_lifetime_seconds
        self._capacity = max_results
        self._shutdown_timeout = shutdown_timeout_seconds
        self._clock = clock
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._shutdown_lock = threading.Lock()
        self._records: dict[str, _Record] = {}
        self._active: tuple[BaseProcess, threading.Thread | None] | None = None
        self._cancel_in_progress = False
        self._closed = False
        self._expiry_thread = threading.Thread(target=self._expire_loop, daemon=True)
        self._expiry_thread.start()

    def submit(
        self,
        audio_path: Path,
        language: Language = "en",
        *,
        replace_oldest_job_id: str | None = None,
    ) -> str:
        if language not in ("en", "it"):
            raise ValueError("Recording language must be 'en' or 'it'.")
        with self._lock:
            if self._closed:
                raise CoordinatorClosedError("The transcription coordinator is closed.")
            self._expire_results()
            if self._active is not None or self._cancel_in_progress:
                raise JobBusyError("A transcription job is already running.")
            oldest_id = next(iter(self._records), None)
            if (
                len(self._records) >= self._capacity
                and replace_oldest_job_id != oldest_id
            ):
                assert oldest_id is not None
                raise ResultCapacityError(oldest_id)

            job_id = uuid.uuid4().hex
            receiver, sender = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_run_job, args=(self._job, audio_path, language, sender)
            )
            try:
                process.start()
            except BaseException:
                receiver.close()
                sender.close()
                raise
            sender.close()
            self._records[job_id] = _Record(JobSnapshot(job_id, "running"))
            monitor = threading.Thread(
                target=self._monitor, args=(job_id, process, receiver), daemon=True
            )
            self._active = (process, monitor)
            try:
                monitor.start()
            except BaseException:
                receiver.close()
                del self._records[job_id]
                self._active = (process, None)
                try:
                    process.terminate()
                    process.join(self._shutdown_timeout)
                    if process.is_alive():
                        process.kill()
                        process.join(self._shutdown_timeout)
                    if process.is_alive():
                        raise RuntimeError("The transcription worker did not stop.")
                except BaseException:
                    self._closed = True
                    self._changed.notify_all()
                    raise
                self._active = None
                raise
            if len(self._records) > self._capacity:
                assert oldest_id is not None
                del self._records[oldest_id]
            return job_id

    def get(self, job_id: str) -> JobSnapshot:
        with self._lock:
            self._expire_results()
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(f"Unknown or expired job: {job_id}")
            return record.snapshot

    def cancel(self, job_id: str) -> JobSnapshot:
        with self._lock:
            if self._closed:
                raise CoordinatorClosedError("The transcription coordinator is closed.")
            self._expire_results()
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(f"Unknown or expired job: {job_id}")
            if record.snapshot.state == "cancelled":
                return record.snapshot
            if record.snapshot.state in ("succeeded", "failed"):
                raise JobFinishedError(f"Job {job_id} has already finished.")
            if self._cancel_in_progress:
                return record.snapshot
            record.snapshot = JobSnapshot(job_id, "cancelling")
            self._cancel_in_progress = True

        try:
            with self._shutdown_lock:
                with self._lock:
                    if self._closed:
                        raise CoordinatorClosedError(
                            "The transcription coordinator is closed."
                        )
                    active = self._active
                if active is not None:
                    process, monitor = active
                    if process.is_alive():
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            if process.is_alive():
                                raise
                    if monitor is not None:
                        monitor.join(self._shutdown_timeout)
                        if monitor.is_alive():
                            if process.is_alive():
                                try:
                                    process.kill()
                                except ProcessLookupError:
                                    if process.is_alive():
                                        raise
                            monitor.join(self._shutdown_timeout)
                        if monitor.is_alive():
                            raise RuntimeError("The transcription worker did not stop.")
                return self.get(job_id)
        finally:
            with self._lock:
                self._cancel_in_progress = False

    def shutdown(self) -> None:
        with self._shutdown_lock:
            with self._lock:
                self._closed = True
                self._changed.notify_all()
                active = self._active
            if active is not None:
                process, monitor = active
                if process.is_alive():
                    process.terminate()
                if monitor is None:
                    process.join(self._shutdown_timeout)
                    still_running = process.is_alive()
                else:
                    monitor.join(self._shutdown_timeout)
                    still_running = monitor.is_alive()
                if still_running:
                    if process.is_alive():
                        process.kill()
                    if monitor is None:
                        process.join(self._shutdown_timeout)
                        still_running = process.is_alive()
                    else:
                        monitor.join(self._shutdown_timeout)
                        still_running = monitor.is_alive()
                if still_running:
                    raise RuntimeError("The transcription worker did not stop.")
            self._expiry_thread.join(self._shutdown_timeout)
            if self._expiry_thread.is_alive():
                raise RuntimeError("The result expiry thread did not stop.")
            with self._lock:
                self._active = None
                self._records.clear()

    def _expire_results(self) -> None:
        now = self._clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.finished_at is not None
            and now - record.finished_at >= self._lifetime
        ]
        for job_id in expired:
            del self._records[job_id]

    def _expire_loop(self) -> None:
        with self._changed:
            while not self._closed:
                self._expire_results()
                deadlines = [
                    record.finished_at + self._lifetime
                    for record in self._records.values()
                    if record.finished_at is not None
                ]
                timeout = (
                    min(threading.TIMEOUT_MAX, max(0, min(deadlines) - self._clock()))
                    if deadlines
                    else None
                )
                self._changed.wait(timeout)

    def _monitor(self, job_id: str, process: BaseProcess, receiver: Connection) -> None:
        try:
            try:
                outcome = receiver.recv()
            except EOFError:
                outcome = None
            except Exception as exc:
                outcome = _Outcome(error=JobError("WorkerCommunicationError", str(exc)))
        finally:
            receiver.close()
            process.join()
        if process.exitcode != 0 or not isinstance(outcome, _Outcome):
            outcome = _Outcome(
                error=JobError(
                    "WorkerExit",
                    f"Transcription worker exited with code {process.exitcode}.",
                )
            )
        with self._lock:
            if not self._closed:
                if self._records[job_id].snapshot.state == "cancelling":
                    snapshot = JobSnapshot(job_id, "cancelled")
                elif outcome.error is not None:
                    snapshot = JobSnapshot(job_id, "failed", error=outcome.error)
                elif isinstance(outcome.result, Transcript):
                    snapshot = JobSnapshot(job_id, "succeeded", result=outcome.result)
                else:
                    snapshot = JobSnapshot(
                        job_id,
                        "failed",
                        error=JobError(
                            "InvalidResult", "Worker returned no transcript."
                        ),
                    )
                self._records[job_id] = _Record(snapshot, self._clock())
                self._changed.notify_all()
            self._active = None
