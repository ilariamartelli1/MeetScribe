"""Exercise the actual Community-1 adapter on two synthetic voices."""

import hashlib
import json
import os
import subprocess
import wave
from collections import Counter
from pathlib import Path

import av
import pytest

from app.community1_diarizer import MODEL_MANIFEST
from app.model_integrity import verify_model

ROOT = Path(__file__).resolve().parents[2]
TWO_VOICES_AUDIO = ROOT / "test/data/smoke/two-voices.wav"
TWO_VOICES_SHA256 = "976ed9a79fd6b456b499643b7d393bc33bc8f29cae5f0fccfc13519e288445e8"
VOICE_INTERVALS_MS = ((0, 7062), (8062, 16062), (17062, 24124), (25124, 33124))
SAMPLE_RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
SECOND_VOICE_SECONDS = 8
SILENCE_SECONDS = 1
MIN_SPEAKER_COVERAGE_MS = 1000
MILLISECONDS_PER_SECOND = 1000
CONTAINER_MEMORY_LIMIT = "6g"
CONTAINER_CPU_LIMIT = "4"
CONTAINER_TIMEOUT_SECONDS = 300


def _mono_pcm(path: Path, limit_seconds: int | None = None) -> bytes:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    pcm = bytearray()
    limit = (
        limit_seconds * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE
        if limit_seconds is not None
        else None
    )
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                pcm.extend(output.to_ndarray().tobytes())
            if limit is not None and len(pcm) >= limit:
                return bytes(pcm[:limit])
        for output in resampler.resample(None):
            pcm.extend(output.to_ndarray().tobytes())
    return bytes(pcm)


def _write_two_voice_fixture(path: Path) -> list[tuple[int, int]]:
    flite = _mono_pcm(ROOT / "test/data/smoke/synthetic-meeting.wav") * 2
    espeak = _mono_pcm(
        ROOT / "test/data/checkpoint/english/meeting-en.wav", SECOND_VOICE_SECONDS
    )
    if (
        not flite
        or len(espeak) != SECOND_VOICE_SECONDS * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE
    ):
        raise ValueError("Synthetic voice source audio is incomplete.")
    silence = b"\x00" * SILENCE_SECONDS * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE
    frames_written = 0
    intervals = []
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(BYTES_PER_SAMPLE)
        audio.setframerate(SAMPLE_RATE_HZ)
        for clip in (flite, espeak, flite, espeak):
            start_ms = frames_written * MILLISECONDS_PER_SECOND // SAMPLE_RATE_HZ
            audio.writeframes(clip)
            frames_written += len(clip) // BYTES_PER_SAMPLE
            intervals.append(
                (start_ms, frames_written * MILLISECONDS_PER_SECOND // SAMPLE_RATE_HZ)
            )
            audio.writeframes(silence)
            frames_written += len(silence) // BYTES_PER_SAMPLE
    return intervals


def _dominant_speaker(turns: list[dict[str, object]], interval: tuple[int, int]) -> str:
    coverage: Counter[str] = Counter()
    start, end = interval
    for turn in turns:
        turn_start = turn["start_ms"]
        turn_end = turn["end_ms"]
        speaker = turn["speaker_id"]
        assert isinstance(turn_start, int)
        assert isinstance(turn_end, int)
        assert isinstance(speaker, str)
        overlap = max(0, min(end, turn_end) - max(start, turn_start))
        coverage[speaker] += overlap
    assert coverage, f"No speaker turns in interval {interval}"
    speaker, covered_ms = coverage.most_common(1)[0]
    assert covered_ms >= MIN_SPEAKER_COVERAGE_MS
    return speaker


def test_versioned_two_voice_fixture_matches_sources(tmp_path: Path) -> None:
    regenerated = tmp_path / "two-voices.wav"
    intervals = _write_two_voice_fixture(regenerated)
    assert tuple(intervals) == VOICE_INTERVALS_MS
    with regenerated.open("rb") as audio:
        assert hashlib.file_digest(audio, "sha256").hexdigest() == TWO_VOICES_SHA256
    with TWO_VOICES_AUDIO.open("rb") as audio:
        assert hashlib.file_digest(audio, "sha256").hexdigest() == TWO_VOICES_SHA256


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run GPU diarization checks.",
)
def test_real_community1_adapter_identifies_two_synthetic_voices(
    asr_image: str,
) -> None:
    with TWO_VOICES_AUDIO.open("rb") as audio:
        assert hashlib.file_digest(audio, "sha256").hexdigest() == TWO_VOICES_SHA256
    model = Path(
        os.environ.get(
            "MEETSCRIBE_DIARIZATION_MODEL_DIR",
            str(ROOT / "models/pyannote-community-1"),
        )
    ).resolve(strict=True)
    verify_model(model, MODEL_MANIFEST)
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--memory",
            CONTAINER_MEMORY_LIMIT,
            "--cpus",
            CONTAINER_CPU_LIMIT,
            "--mount",
            f"type=bind,src={model},dst=/diarization,readonly",
            "--mount",
            f"type=bind,src={TWO_VOICES_AUDIO},dst=/audio/two-voices.wav,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/diarization_adapter_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"Community-1 adapter failed:\n{completed.stderr}\n{completed.stdout}"
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["provenance"]["device"] == "cuda"
    assert result["turns"] and result["exclusive_turns"]
    speakers = [
        _dominant_speaker(result["turns"], interval) for interval in VOICE_INTERVALS_MS
    ]
    assert speakers[0] == speakers[2] and speakers[1] == speakers[3]
    assert speakers[0] != speakers[1], f"Distinct voices were not separated: {speakers}"
