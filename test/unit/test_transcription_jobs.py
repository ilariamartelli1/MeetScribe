"""Verify the job coordinator: process management, result retention, worker failures and shutdown."""

import os
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from app.asr_contract import Language, ModelProvenance, Transcript, TranscriptSegment
from app.transcription_jobs import (
    CoordinatorClosedError,
    JobBusyError,
    JobNotFoundError,
    ResultCapacityError,
    TranscriptionCoordinator,
)

PROVENANCE = ModelProvenance(
    "fake", "1.0", "model", "revision", "digest", "cpu", "int8", True, 5
)


@dataclass(frozen=True)
class FakeJob:
    directory: Path
    mode: str = "success"

    def __call__(self, audio_path: Path, language: Language) -> Transcript:
        if self.mode == "ignore_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (self.directory / "started").write_text(str(os.getpid()))
        (self.directory / f"started-{audio_path.name}").touch()
        if self.mode in ("wait", "ignore_term"):
            while not (self.directory / "release").exists():
                time.sleep(0.01)
        if self.mode == "error":
            raise ValueError("Synthetic inference failure")
        if self.mode == "crash":
            os._exit(17)
        return Transcript(
            language, 1000, (TranscriptSegment(0, 100, audio_path.name),), PROVENANCE
        )


def wait_for(predicate: Callable[[], bool], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for the worker.")
        time.sleep(0.01)


def test_running_job_is_pollable_and_rejects_concurrent_work(tmp_path: Path) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
    try:
        job_id = coordinator.submit(Path("first.wav"), "it")
        wait_for(lambda: (tmp_path / "started").exists())
        assert coordinator.get(job_id).state == "running"
        with pytest.raises(JobBusyError):
            coordinator.submit(Path("second.wav"))
        (tmp_path / "release").touch()
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")
        result = coordinator.get(job_id).result
        assert result is not None
        assert result.language == "it"
        assert result.segments[0].text == "first.wav"
        next_id = coordinator.submit(Path("second.wav"))
        wait_for(lambda: coordinator.get(next_id).state == "succeeded")
    finally:
        coordinator.shutdown()


@pytest.mark.parametrize(
    ("mode", "kind", "message"),
    [
        ("error", "ValueError", "Synthetic inference failure"),
        ("crash", "WorkerExit", "code 17"),
    ],
)
def test_failures_are_reported_and_release_the_job_slot(
    tmp_path: Path, mode: str, kind: str, message: str
) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, mode))
    try:
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "failed")
        error = coordinator.get(job_id).error
        assert error is not None
        assert error.kind == kind
        assert message in error.message
        next_id = coordinator.submit(Path("another.wav"))
        wait_for(lambda: coordinator.get(next_id).state == "failed")
    finally:
        coordinator.shutdown()


def test_capacity_requires_confirmation_of_the_actual_oldest_job(
    tmp_path: Path,
) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path))
    try:
        jobs = []
        for index in range(16):
            job_id = coordinator.submit(Path(f"{index}.wav"))
            wait_for(lambda: coordinator.get(job_id).state == "succeeded")
            jobs.append(job_id)
        first, second = jobs[:2]

        with pytest.raises(ResultCapacityError) as full:
            coordinator.submit(Path("17th.wav"))
        assert full.value.oldest_job_id == first
        assert not (tmp_path / "started-17th.wav").exists()
        with pytest.raises(ResultCapacityError):
            coordinator.submit(Path("17th.wav"), replace_oldest_job_id=second)
        assert coordinator.get(first).result is not None
        third = coordinator.submit(Path("17th.wav"), replace_oldest_job_id=first)
        with pytest.raises(JobNotFoundError):
            coordinator.get(first)
        assert coordinator.get(second).result is not None
        wait_for(lambda: coordinator.get(third).state == "succeeded")
    finally:
        coordinator.shutdown()


def test_results_expire_one_hour_after_completion(tmp_path: Path) -> None:
    now = [0.0]
    coordinator = TranscriptionCoordinator(
        FakeJob(tmp_path), max_results=1, clock=lambda: now[0]
    )
    try:
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")
        now[0] = 3599
        assert coordinator.get(job_id).state == "succeeded"
        now[0] = 3600
        with pytest.raises(JobNotFoundError):
            coordinator.get(job_id)
        next_id = coordinator.submit(Path("next.wav"))
        wait_for(lambda: coordinator.get(next_id).state == "succeeded")
    finally:
        coordinator.shutdown()


def test_expired_result_is_removed_without_polling(tmp_path: Path) -> None:
    coordinator = TranscriptionCoordinator(
        FakeJob(tmp_path), result_lifetime_seconds=0.1
    )
    try:
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")

        def removed() -> bool:
            with coordinator._lock:
                return job_id not in coordinator._records

        wait_for(removed)
        with pytest.raises(JobNotFoundError):
            coordinator.get(job_id)
    finally:
        coordinator.shutdown()


def test_failed_process_start_does_not_replace_a_saved_result(tmp_path: Path) -> None:
    job = FakeJob(tmp_path)
    coordinator = TranscriptionCoordinator(job, max_results=1)
    try:
        first = coordinator.submit(Path("first.wav"))
        wait_for(lambda: coordinator.get(first).state == "succeeded")
        coordinator._job = lambda path, language: job(path, language)
        with pytest.raises((AttributeError, TypeError)):
            coordinator.submit(Path("second.wav"), replace_oldest_job_id=first)
        assert coordinator.get(first).state == "succeeded"
        coordinator._job = job
        second = coordinator.submit(Path("second.wav"), replace_oldest_job_id=first)
        wait_for(lambda: coordinator.get(second).state == "succeeded")
    finally:
        coordinator.shutdown()


def test_failed_monitor_start_stops_worker_and_preserves_saved_result(
    tmp_path: Path,
) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path), max_results=1)
    try:
        first = coordinator.submit(Path("first.wav"))
        wait_for(lambda: coordinator.get(first).state == "succeeded")
        with (
            patch(
                "app.transcription_jobs.threading.Thread.start",
                side_effect=RuntimeError("Cannot start monitor"),
            ),
            pytest.raises(RuntimeError, match="Cannot start monitor"),
        ):
            coordinator.submit(Path("second.wav"), replace_oldest_job_id=first)
        assert coordinator.get(first).state == "succeeded"
        second = coordinator.submit(Path("second.wav"), replace_oldest_job_id=first)
        wait_for(lambda: coordinator.get(second).state == "succeeded")
    finally:
        coordinator.shutdown()


def test_shutdown_can_retry_if_worker_cleanup_initially_fails(tmp_path: Path) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
    try:
        with (
            patch(
                "app.transcription_jobs.threading.Thread.start",
                side_effect=RuntimeError("Cannot start monitor"),
            ),
            patch(
                "multiprocessing.process.BaseProcess.terminate",
                side_effect=OSError("Cannot terminate worker"),
            ),
            pytest.raises(OSError, match="Cannot terminate worker"),
        ):
            coordinator.submit(Path("speech.wav"))
        with pytest.raises(CoordinatorClosedError):
            coordinator.submit(Path("another.wav"))
        assert coordinator._active is not None
        process, _ = coordinator._active
        coordinator.shutdown()
        assert not process.is_alive()
    finally:
        coordinator.shutdown()


def test_concurrent_submissions_start_only_one_worker(tmp_path: Path) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
    barrier = threading.Barrier(2)

    def submit(name: str) -> str:
        barrier.wait()
        try:
            return coordinator.submit(Path(name))
        except JobBusyError:
            return "busy"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(submit, "first.wav")
            second = pool.submit(submit, "second.wav")
            results = [first.result(), second.result()]
        assert results.count("busy") == 1
        job_id = next(result for result in results if result != "busy")
        wait_for(lambda: (tmp_path / "started").exists())
        assert coordinator.get(job_id).state == "running"
    finally:
        coordinator.shutdown()


def test_shutdown_kills_child_that_ignores_termination(tmp_path: Path) -> None:
    coordinator = TranscriptionCoordinator(
        FakeJob(tmp_path, "ignore_term"), shutdown_timeout_seconds=0.2
    )
    try:
        coordinator.submit(Path("speech.wav"))
        wait_for(lambda: (tmp_path / "started").exists())
        child_pid = int((tmp_path / "started").read_text())
        coordinator.shutdown()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        coordinator.shutdown()


def test_shutdown_stops_child_discards_results_and_rejects_new_jobs(
    tmp_path: Path,
) -> None:
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
    job_id = coordinator.submit(Path("speech.wav"))
    wait_for(lambda: (tmp_path / "started").exists())
    child_pid = int((tmp_path / "started").read_text())
    try:
        coordinator.shutdown()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with pytest.raises(JobNotFoundError):
            coordinator.get(job_id)
        with pytest.raises(CoordinatorClosedError):
            coordinator.submit(Path("new.wav"))
        coordinator.shutdown()
    finally:
        coordinator.shutdown()


def test_invalid_settings_and_languages_fail_explicitly(tmp_path: Path) -> None:
    for settings in (
        {"max_results": 0},
        {"result_lifetime_seconds": 0},
        {"result_lifetime_seconds": float("nan")},
        {"shutdown_timeout_seconds": float("inf")},
    ):
        with pytest.raises(ValueError):
            TranscriptionCoordinator(FakeJob(tmp_path), **settings)
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path))
    try:
        with pytest.raises(ValueError, match="language"):
            coordinator.submit(Path("speech.wav"), "fr")  # type: ignore[arg-type]
        with pytest.raises(JobNotFoundError):
            coordinator.get("unknown")
    finally:
        coordinator.shutdown()
