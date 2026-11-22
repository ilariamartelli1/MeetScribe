"""Verify health and temporary job endpoints."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.core.minutes_contract import MinutesDraft
from app.core.minutes_extractor import MinutesExtractionResult
from app.core.speaker_alignment import AttributedTranscript
from app.core.transcription_jobs import TranscriptionCoordinator
from test.unit.test_transcription_jobs import FakeJob, FakeSpeakerJob, wait_for


class FakeMinutesExtractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self, transcript: AttributedTranscript
    ) -> MinutesExtractionResult:
        self.calls += 1
        return MinutesExtractionResult(
            minutes=MinutesDraft(
                summary="The meeting covered a plan.",
                topics=(),
                decisions=(),
                action_items=(),
            ),
            included_segment_ids=("segment-1",),
            context_size=4096,
            prompt_token_upper_bound=256,
            prompt_tokens=128,
            generated_tokens=64,
        )


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_endpoint_is_not_healthy() -> None:
    response = TestClient(app).get("/api/v1/missing")

    assert response.status_code == 404


def test_job_status_hides_transcript_and_explicit_download_returns_it(
    tmp_path: Path,
) -> None:
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        coordinator = TranscriptionCoordinator(FakeSpeakerJob(tmp_path))
        app.state.coordinator = coordinator
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["state"] == "succeeded"
        assert "result" not in response.json()
        assert "hello" not in response.text
        assert response.headers["cache-control"] == "no-store"

        transcript = client.get(f"/api/v1/jobs/{job_id}/transcript.txt")
        assert transcript.status_code == 200
        assert "attachment" in transcript.headers["content-disposition"]
        assert "Voice 1: hello" in transcript.text
        assert transcript.headers["cache-control"] == "no-store"


def test_minutes_api_edit_review_and_exports_are_revision_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "create_ollama_minutes_extractor", lambda _client: FakeMinutesExtractor()
    )
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        coordinator = TranscriptionCoordinator(FakeSpeakerJob(tmp_path))
        app.state.coordinator = coordinator
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")
        origin = {"origin": "http://localhost"}

        started = client.post(f"/api/v1/jobs/{job_id}/minutes", headers=origin)
        assert started.status_code == 202
        assert started.json()["state"] == "running"
        minutes_url = f"/api/v1/jobs/{job_id}/minutes"
        wait_for(lambda: client.get(minutes_url).json()["state"] == "succeeded")
        snapshot = client.get(minutes_url)
        assert snapshot.status_code == 200
        assert snapshot.json()["minutes"]["summary"] == "The meeting covered a plan."
        assert snapshot.json()["extraction_metrics"] == {
            "context_size": 4096,
            "prompt_token_upper_bound": 256,
            "prompt_tokens": 128,
            "generated_tokens": 64,
        }

        edited_minutes = snapshot.json()["minutes"]
        edited_minutes["summary"] = "La città rivedrà il piano."
        edited_minutes["action_items"] = [
            {
                "description": "Ask Marta to review the plan",
                "owner_name": "Marta",
                "owner_speaker_id": None,
                "deadline_text": None,
                "deadline_date": None,
                "source_segment_ids": [],
            }
        ]
        edited = client.patch(
            minutes_url,
            headers=origin,
            json={"expected_revision": 1, "minutes": edited_minutes},
        )
        assert edited.status_code == 200
        assert edited.json()["minutes_revision"] == 2
        assert edited.json()["review_status"] == "DRAFT / UNREVIEWED"
        assert "hello" not in edited.text
        assert (
            client.patch(
                minutes_url,
                json={"expected_revision": 2, "minutes": edited_minutes},
            ).status_code
            == 403
        )

        conflict = client.post(
            f"{minutes_url}/review",
            headers=origin,
            json={"expected_revision": 1},
        )
        assert conflict.status_code == 409
        reviewed = client.post(
            f"{minutes_url}/review",
            headers=origin,
            json={"expected_revision": 2},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["review_status"] == "REVIEWED"

        text_export = client.get(f"{minutes_url}.txt")
        assert text_export.status_code == 200
        assert text_export.text.startswith("REVIEWED\n")
        assert "La città rivedrà il piano." in text_export.text
        assert "Marta" in text_export.text
        assert "hello" not in text_export.text

        json_export = client.get(f"{minutes_url}.json")
        assert json_export.status_code == 200
        payload = json_export.json()
        assert payload["schema_version"] == 1
        assert payload["review_status"] == "REVIEWED"
        assert payload["minutes_revision"] == 2
        assert payload["segments"][0]["text"] == "hello"
        assert payload["minutes"]["summary"] == "La città rivedrà il piano."


def test_minutes_api_requires_confirmation_for_regeneration_after_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor = FakeMinutesExtractor()
    monkeypatch.setattr(
        main, "create_ollama_minutes_extractor", lambda _client: extractor
    )
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        coordinator = TranscriptionCoordinator(FakeSpeakerJob(tmp_path))
        app.state.coordinator = coordinator
        job_id = coordinator.submit(Path("speech.wav"))
        wait_for(lambda: coordinator.get(job_id).state == "succeeded")
        origin = {"origin": "http://localhost"}
        minutes_url = f"/api/v1/jobs/{job_id}/minutes"

        assert client.post(minutes_url, headers=origin).status_code == 202
        wait_for(lambda: client.get(minutes_url).json()["state"] == "succeeded")
        current = client.get(minutes_url).json()
        edited_minutes = current["minutes"]
        edited_minutes["summary"] = "A human-edited summary."
        edited = client.patch(
            minutes_url,
            headers=origin,
            json={"expected_revision": 1, "minutes": edited_minutes},
        )
        assert edited.status_code == 200
        assert edited.json()["minutes_revision"] == 2

        regenerate_url = f"{minutes_url}/regenerate"
        request_body = {"expected_revision": 2}
        assert (
            client.post(regenerate_url, headers=origin, json=request_body).status_code
            == 409
        )
        response = client.post(
            regenerate_url,
            headers=origin,
            json={**request_body, "confirm_overwrite": True},
        )
        assert response.status_code == 202
        wait_for(
            lambda: (
                (snapshot := client.get(minutes_url).json())["state"] == "succeeded"
                and not snapshot["regenerating"]
                and snapshot["minutes_revision"] == 3
            )
        )
        regenerated = client.get(minutes_url).json()
        assert regenerated["minutes"]["summary"] == "The meeting covered a plan."
        assert regenerated["review_status"] == "DRAFT / UNREVIEWED"
        assert extractor.calls == 2


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
