"""Verify health and temporary job endpoints."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.transcription_jobs import TranscriptionCoordinator
from test.unit.test_transcription_jobs import FakeJob, wait_for


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_endpoint_is_not_healthy() -> None:
    response = TestClient(app).get("/api/v1/missing")

    assert response.status_code == 404


def test_cancel_endpoint_and_status_remain_responsive(tmp_path: Path) -> None:
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
        app.state.coordinator = coordinator
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: (tmp_path / "started").exists())
        url = f"/api/v1/jobs/{job_id}"
        headers = {"origin": "http://localhost"}

        assert client.post(f"{url}/cancel").status_code == 403
        assert (
            client.post(f"{url}/cancel", headers={"origin": "http://evil"}).status_code
            == 403
        )
        assert client.get(url, headers={"host": "evil.example"}).status_code == 403
        assert coordinator.get(job_id).state == "running"

        assert coordinator._active is not None
        process, _ = coordinator._active
        terminate = process.terminate
        entered = threading.Event()
        release = threading.Event()

        def delayed_terminate() -> None:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Cancellation test did not release the worker.")
            terminate()

        with patch.object(process, "terminate", side_effect=delayed_terminate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                cancelled = pool.submit(client.post, f"{url}/cancel", headers=headers)
                try:
                    assert entered.wait(5)
                    status = pool.submit(client.get, url).result(timeout=5)
                    assert status.json()["state"] == "cancelling"
                    assert status.headers["cache-control"] == "no-store"
                    assert (
                        pool.submit(client.post, f"{url}/cancel", headers=headers)
                        .result(timeout=5)
                        .json()["state"]
                        == "cancelling"
                    )
                finally:
                    release.set()
                response = cancelled.result(timeout=5)

        assert response.status_code == 200
        assert response.json()["state"] == "cancelled"
        assert response.headers["cache-control"] == "no-store"
        assert (
            client.post(f"{url}/cancel", headers=headers).json()["state"] == "cancelled"
        )
        assert (
            client.post("/api/v1/jobs/unknown/cancel", headers=headers).status_code
            == 404
        )

        (tmp_path / "release").touch()
        finished = coordinator.submit(Path("finished.wav"))
        wait_for(lambda: coordinator.get(finished).state == "succeeded")
        response = client.post(f"/api/v1/jobs/{finished}/cancel", headers=headers)
        assert response.status_code == 409
        assert coordinator.get(finished).result is not None


def test_cancel_endpoint_reports_stop_failure_and_allows_retry(tmp_path: Path) -> None:
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
        app.state.coordinator = coordinator
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: (tmp_path / "started").exists())
        assert coordinator._active is not None
        process, _ = coordinator._active
        url = f"/api/v1/jobs/{job_id}/cancel"
        headers = {"origin": "http://localhost"}
        with patch.object(process, "terminate", side_effect=OSError("Stop failed")):
            response = client.post(url, headers=headers)
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert coordinator.get(job_id).state == "cancelling"
        assert client.post(url, headers=headers).json()["state"] == "cancelled"
