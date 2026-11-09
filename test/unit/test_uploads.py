"""Verify bounded uploads, audio validation, and managed-file cleanup."""

import io
import struct
import threading
import wave
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.transcription_jobs import TranscriptionCoordinator
from app.uploads import (
    MAX_AUDIO_BYTES,
    MAX_REQUEST_BYTES,
    UploadError,
    UploadParser,
    ValidationWorkerStuck,
    validate_audio,
)
from test.unit.test_transcription_jobs import FakeJob, wait_for

HEADERS = {"origin": "http://localhost"}
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "asr"


def wav_bytes(samples: int = 800, sample_rate: int = 8000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * samples)
    return output.getvalue()


def stereo_speech_bytes(mode: str) -> bytes:
    with wave.open(str(FIXTURES / "synthetic-meeting.wav"), "rb") as source:
        samples = source.readframes(source.getnframes())
    frames = bytearray()
    for (sample,) in struct.iter_unpack("<h", samples):
        if mode == "left":
            left, right = sample, 0
        elif mode == "right":
            left, right = 0, sample
        else:
            left, right = sample, max(-32768, min(32767, -sample))
        frames.extend(struct.pack("<hh", left, right))
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(frames)
    return output.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(app, base_url="http://localhost") as test_client:
        app.state.coordinator.shutdown()
        app.state.coordinator = TranscriptionCoordinator(FakeJob(tmp_path))
        yield test_client


def test_upload_runs_job_and_removes_generated_audio(client: TestClient) -> None:
    response = client.post(
        "/api/v1/jobs",
        headers=HEADERS,
        files={"file": ("../../caller-name.wav", wav_bytes(), "audio/wav")},
        data={"language": "it"},
    )
    assert response.status_code == 202
    assert response.headers["cache-control"] == "no-store"
    job_id = response.json()["job_id"]
    coordinator = app.state.coordinator
    wait_for(lambda: coordinator.get(job_id).state == "succeeded")
    result = coordinator.get(job_id).result
    assert result is not None
    assert result.language == "it"
    assert result.segments[0].text.startswith("upload-")
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.json()["state"] == "succeeded"
    assert status.json()["result"]["language"] == "it"
    assert status.headers["cache-control"] == "no-store"
    assert not list(app.state.upload_dir.iterdir())


def test_mp3_upload_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/jobs",
        headers=HEADERS,
        files={
            "file": ("sample.mp3", (FIXTURES / "synthetic-meeting.mp3").read_bytes())
        },
    )
    assert response.status_code == 202, response.text
    wait_for(
        lambda: app.state.coordinator.get(response.json()["job_id"]).state
        == "succeeded"
    )
    assert not list(app.state.upload_dir.iterdir())


@pytest.mark.parametrize(
    ("name", "audio", "message"),
    [
        ("bad.wav", b"not audio", "corrupt"),
        ("empty.wav", b"", "nonempty"),
        ("no-samples.wav", wav_bytes(samples=0), "samples"),
        ("truncated.wav", wav_bytes()[:-100], "corrupt"),
        ("wrong.mp3", wav_bytes(), "match"),
        ("wrong.wav", (FIXTURES / "synthetic-meeting.mp3").read_bytes(), "match"),
        ("bad.ogg", b"data", "WAV and MP3"),
        ("left.wav", stereo_speech_bytes("left"), "mono"),
        ("right.wav", stereo_speech_bytes("right"), "mono"),
        ("opposed.wav", stereo_speech_bytes("opposed"), "mono"),
        ("low-rate.wav", wav_bytes(sample_rate=4000), "sample rate"),
        ("high-rate.wav", wav_bytes(sample_rate=96000), "sample rate"),
        pytest.param(
            "long.wav",
            wav_bytes(samples=300 * 8000 + 1),
            "five minutes",
            id="too-long",
        ),
    ],
    ids=[
        "corrupt",
        "empty",
        "no-samples",
        "truncated",
        "wav-as-mp3",
        "mp3-as-wav",
        "extension",
        "left-only",
        "right-only",
        "opposite-phase",
        "low-rate",
        "high-rate",
        "too-long",
    ],
)
def test_invalid_audio_is_rejected_before_start(
    client: TestClient, tmp_path: Path, name: str, audio: bytes, message: str
) -> None:
    response = client.post(
        "/api/v1/jobs", headers=HEADERS, files={"file": (name, audio)}
    )
    assert response.status_code == 422, response.text
    assert message in response.json()["detail"]
    assert response.headers["cache-control"] == "no-store"
    assert not (tmp_path / "started").exists()
    assert not list(app.state.upload_dir.iterdir())


def test_exactly_five_minutes_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/jobs",
        headers=HEADERS,
        files={"file": ("boundary.wav", wav_bytes(samples=300 * 8000))},
    )
    assert response.status_code == 202, response.text
    wait_for(
        lambda: app.state.coordinator.get(response.json()["job_id"]).state
        == "succeeded"
    )


def test_replacement_requires_explicit_current_confirmation(
    client: TestClient, tmp_path: Path
) -> None:
    app.state.coordinator.shutdown()
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path), max_results=1)
    app.state.coordinator = coordinator
    files = {"file": ("speech.wav", wav_bytes())}
    first = client.post("/api/v1/jobs", headers=HEADERS, files=files).json()["job_id"]
    wait_for(lambda: coordinator.get(first).state == "succeeded")
    markers = set(tmp_path.glob("started-*"))
    full = client.post("/api/v1/jobs", headers=HEADERS, files=files)
    assert full.status_code == 409
    assert full.json()["oldest_job_id"] == first
    assert coordinator.get(first).result is not None
    assert set(tmp_path.glob("started-*")) == markers
    assert not list(app.state.upload_dir.iterdir())
    stale = client.post(
        "/api/v1/jobs",
        headers=HEADERS,
        files=files,
        data={"replace_oldest_job_id": "0" * 32},
    )
    assert stale.status_code == 409
    assert stale.json()["oldest_job_id"] == first
    confirmed = client.post(
        "/api/v1/jobs",
        headers=HEADERS,
        files=files,
        data={"replace_oldest_job_id": first},
    )
    assert confirmed.status_code == 202
    assert client.get(f"/api/v1/jobs/{first}").status_code == 404
    wait_for(lambda: coordinator.get(confirmed.json()["job_id"]).state == "succeeded")
    assert not list(app.state.upload_dir.iterdir())


@pytest.mark.parametrize("mode", ["error", "wait"])
def test_audio_is_removed_after_failure_or_cancel(
    client: TestClient, tmp_path: Path, mode: str
) -> None:
    app.state.coordinator.shutdown()
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, mode))
    app.state.coordinator = coordinator
    response = client.post(
        "/api/v1/jobs", headers=HEADERS, files={"file": ("speech.wav", wav_bytes())}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    if mode == "wait":
        wait_for(lambda: (tmp_path / "started").exists())
        assert list(app.state.upload_dir.iterdir())
        response = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=HEADERS)
        assert response.json()["state"] == "cancelled"
    else:
        wait_for(lambda: coordinator.get(job_id).state == "failed")
    assert not list(app.state.upload_dir.iterdir())


def test_busy_job_rejects_another_upload_and_removes_only_new_audio(
    client: TestClient, tmp_path: Path
) -> None:
    app.state.coordinator.shutdown()
    coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
    app.state.coordinator = coordinator
    files = {"file": ("speech.wav", wav_bytes())}
    first = client.post("/api/v1/jobs", headers=HEADERS, files=files)
    assert first.status_code == 202
    wait_for(lambda: (tmp_path / "started").exists())
    paths = list(app.state.upload_dir.iterdir())
    assert len(paths) == 1
    second = client.post("/api/v1/jobs", headers=HEADERS, files=files)
    assert second.status_code == 409
    assert list(app.state.upload_dir.iterdir()) == paths
    assert (
        client.post(
            f"/api/v1/jobs/{first.json()['job_id']}/cancel", headers=HEADERS
        ).json()["state"]
        == "cancelled"
    )
    assert not list(app.state.upload_dir.iterdir())


def test_decoder_timeout_rejects_and_cleans_audio(
    client: TestClient, tmp_path: Path
) -> None:
    with patch("app.uploads.DECODE_TIMEOUT_SECONDS", 0.001):
        response = client.post(
            "/api/v1/jobs", headers=HEADERS, files={"file": ("speech.wav", wav_bytes())}
        )
    assert response.status_code == 422
    assert "timed out" in response.json()["detail"]
    assert not (tmp_path / "started").exists()
    assert not list(app.state.upload_dir.iterdir())


def test_missing_temporary_audio_is_reported_as_server_failure(tmp_path: Path) -> None:
    with pytest.raises(UploadError) as error:
        validate_audio(tmp_path / "missing.wav", "wav")
    assert error.value.status_code == 503


def test_stuck_decoder_blocks_uploads_and_is_retried_at_shutdown(
    tmp_path: Path,
) -> None:
    process = Mock()
    with (
        patch("app.main.validate_audio", side_effect=ValidationWorkerStuck(process)),
        patch("app.main.stop_validation_worker") as stop,
    ):
        with TestClient(app, base_url="http://localhost") as client:
            app.state.coordinator.shutdown()
            app.state.coordinator = TranscriptionCoordinator(FakeJob(tmp_path))
            upload_dir = app.state.upload_dir
            files = {"file": ("speech.wav", wav_bytes())}
            response = client.post("/api/v1/jobs", headers=HEADERS, files=files)
            assert response.status_code == 503
            assert len(list(upload_dir.iterdir())) == 1
            assert (
                client.post("/api/v1/jobs", headers=HEADERS, files=files).status_code
                == 503
            )
        stop.assert_called_once_with(process)
        assert not upload_dir.exists()


def test_shutdown_removes_audio_of_running_job(tmp_path: Path) -> None:
    with TestClient(app, base_url="http://localhost") as client:
        app.state.coordinator.shutdown()
        app.state.coordinator = TranscriptionCoordinator(FakeJob(tmp_path, "wait"))
        upload_dir = app.state.upload_dir
        response = client.post(
            "/api/v1/jobs", headers=HEADERS, files={"file": ("speech.wav", wav_bytes())}
        )
        assert response.status_code == 202
        wait_for(lambda: (tmp_path / "started").exists())
        assert list(upload_dir.iterdir())
    assert not upload_dir.exists()


def test_failed_upload_cleanup_blocks_uploads_until_shutdown(
    client: TestClient,
) -> None:
    with patch.object(UploadParser, "discard", side_effect=OSError("Deletion failed")):
        response = client.post(
            "/api/v1/jobs",
            headers=HEADERS,
            files={"file": ("bad.wav", b"invalid")},
        )
    assert response.status_code == 503
    assert "cleanup failed" in response.json()["detail"]
    assert len(list(app.state.upload_dir.iterdir())) == 1
    assert (
        client.post(
            "/api/v1/jobs",
            headers=HEADERS,
            files={"file": ("speech.wav", wav_bytes())},
        ).status_code
        == 503
    )


def test_request_validation_and_origin(client: TestClient, tmp_path: Path) -> None:
    body = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
    )
    body += wav_bytes() + b"\r\n--x--\r\n"
    headers = HEADERS | {"content-type": "multipart/form-data; boundary=x"}
    request = client.build_request(
        "POST", "/api/v1/jobs", headers=headers, content=body
    )
    del request.headers["content-length"]
    assert client.send(request).status_code == 411
    assert (
        client.post("/api/v1/jobs", headers=headers, content=body[:-3]).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/jobs",
            headers=headers | {"content-type": "a" * 8193},
            content=b"",
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/v1/jobs",
            headers=headers | {"content-length": str(len(body) + 1)},
            content=body,
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/jobs",
            headers=headers | {"content-length": str(len(body) - 1)},
            content=body,
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/v1/jobs",
            headers=headers | {"content-length": str(MAX_REQUEST_BYTES + 1)},
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/v1/jobs", headers=headers | {"content-length": "wrong"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/jobs",
            content=body,
            headers={"content-type": headers["content-type"]},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/jobs", content=body, headers=headers | {"origin": "http://evil"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/jobs", content=body, headers=headers | {"host": "evil.example"}
        ).status_code
        == 403
    )
    assert not (tmp_path / "started").exists()
    assert not list(app.state.upload_dir.iterdir())


def test_multipart_abuse_is_rejected(client: TestClient) -> None:
    headers = HEADERS | {"content-type": "multipart/form-data; boundary=x"}
    file_part = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
    )
    file_part += wav_bytes() + b"\r\n"
    field = b'--x\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n'
    confirmation = (
        b'--x\r\nContent-Disposition: form-data; name="replace_oldest_job_id"\r\n\r\n'
    )
    confirmation += b"0" * 32 + b"\r\n"
    for body in (
        file_part + file_part + b"--x--\r\n",
        file_part + field + field + b"--x--\r\n",
        file_part + field + field + field + b"--x--\r\n",
        file_part + field + confirmation + field + b"--x--\r\n",
        file_part
        + b'--x\r\nContent-Disposition: form-data; name="language"\r\n\r\n'
        + b"a" * 1025
        + b"\r\n--x--\r\n",
        b"--x\r\nX-Long: " + b"a" * 8193 + b"\r\n\r\nx\r\n--x--\r\n",
    ):
        response = client.post("/api/v1/jobs", headers=headers, content=body)
        assert response.status_code in (400, 413), response.text
        assert not list(app.state.upload_dir.iterdir())


@pytest.mark.parametrize("extra", [0, 1])
def test_exact_audio_size_limit(tmp_path: Path, extra: int) -> None:
    parser = UploadParser("multipart/form-data; boundary=x", tmp_path)
    parser._file.close()
    parser._file = Mock()
    try:
        parser.write(
            b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        )
        block = b"\0" * 1_000_000
        for _ in range(100):
            parser.write(block)
        assert MAX_AUDIO_BYTES == 100_000_000
        if extra:
            with pytest.raises(UploadError, match="100 MB"):
                parser.write(b"\0\r\n--x--\r\n")
        else:
            parser.write(b"\r\n--x--\r\n")
            assert parser.finish()[3] == "wav"
    finally:
        parser.discard()


def test_concurrent_uploads_are_bounded_and_status_remains_responsive(
    client: TestClient,
) -> None:
    entered, release = threading.Event(), threading.Event()

    def wait_to_validate(path: Path, audio_format: str) -> None:
        entered.set()
        assert release.wait(10)

    with patch("app.main.validate_audio", side_effect=wait_to_validate):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                client.post,
                "/api/v1/jobs",
                headers=HEADERS,
                files={"file": ("speech.wav", wav_bytes())},
            )
            try:
                assert entered.wait(5)
                response = pool.submit(
                    client.post,
                    "/api/v1/jobs",
                    headers=HEADERS,
                    files={"file": ("other.wav", wav_bytes())},
                ).result(timeout=5)
                assert response.status_code == 409
                assert (
                    pool.submit(client.get, "/api/v1/jobs/unknown")
                    .result(timeout=5)
                    .status_code
                    == 404
                )
            finally:
                release.set()
            response = first.result(timeout=10)
    assert response.status_code == 202
    wait_for(
        lambda: app.state.coordinator.get(response.json()["job_id"]).state
        == "succeeded"
    )
    assert not list(app.state.upload_dir.iterdir())
