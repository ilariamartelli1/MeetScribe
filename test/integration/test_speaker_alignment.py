"""Check attributed transcription using two synthetic voices and offline models."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from test.integration.asr_model_config import asr_model_mount_args
from test.integration.test_asr_accuracy import normalize_text
from test.src.audio_fixtures import VOICE_INTERVALS_MS

ROOT = Path(__file__).resolve().parents[2]
MIN_INTERVAL_COVERAGE = 0.8
MIN_TRANSCRIBED_COVERAGE_MS = 1000


def _covered_ms(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    last_end = 0
    for start, end in sorted(intervals):
        covered += max(0, end - max(start, last_end))
        last_end = max(last_end, end)
    return covered


def interval_speaker(
    interval: tuple[int, int], attributions: list[dict[str, object]]
) -> str:
    start, end = interval
    all_intervals = []
    speaker_intervals: dict[str, list[tuple[int, int]]] = {}
    for entry in attributions:
        entry_start, entry_end, speaker = (
            entry["start_ms"],
            entry["end_ms"],
            entry["speaker"],
        )
        assert isinstance(entry_start, int) and isinstance(entry_end, int)
        assert speaker is None or isinstance(speaker, str)
        overlap = max(start, entry_start), min(end, entry_end)
        if overlap[0] >= overlap[1]:
            continue
        all_intervals.append(overlap)
        if speaker is not None:
            speaker_intervals.setdefault(speaker, []).append(overlap)
    total = _covered_ms(all_intervals)
    assert total >= MIN_TRANSCRIBED_COVERAGE_MS, (interval, attributions)
    speakers = [
        speaker
        for speaker, intervals in speaker_intervals.items()
        if _covered_ms(intervals) >= total * MIN_INTERVAL_COVERAGE
    ]
    assert len(speakers) == 1, (interval, attributions)
    return speakers[0]


@pytest.fixture(scope="module")
def attributed_job_report(asr_image: str) -> str:
    diarization_model = Path(
        os.environ.get(
            "MEETSCRIBE_DIARIZATION_MODEL_DIR",
            str(ROOT / "models/pyannote-community-1"),
        )
    ).resolve(strict=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--memory",
            "6g",
            "--cpus",
            "4",
            *asr_model_mount_args("/asr"),
            "--mount",
            f"type=bind,src={diarization_model},dst=/diarization,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/data/smoke'},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/speaker_alignment_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=540,
    )
    assert result.returncode == 0, (
        f"Speaker job failed:\n{result.stderr}\n{result.stdout}"
    )
    return result.stdout


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_offline_job_attributes_alternating_synthetic_voices(
    attributed_job_report: str,
) -> None:
    report = json.loads(attributed_job_report)["en"]
    assert report["provenance"]["device"] == "cuda"
    assert report["diarization"]["provenance"]["device"] == "cuda"
    segments = report["segments"]
    attributions = report["attributions"]
    assert segments and len(segments) == len(attributions)
    assert all(
        (segment["start_ms"], segment["end_ms"])
        == (attribution["start_ms"], attribution["end_ms"])
        for segment, attribution in zip(segments, attributions, strict=True)
    )
    interval_speakers = [
        interval_speaker(interval, attributions) for interval in VOICE_INTERVALS_MS
    ]
    assert interval_speakers[0] == interval_speakers[2]
    assert interval_speakers[1] == interval_speakers[3]
    assert interval_speakers[0] != interval_speakers[1]


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_offline_job_produces_segments_and_turns_in_english_and_italian(
    attributed_job_report: str,
) -> None:
    reports = json.loads(attributed_job_report)
    for language in ("en", "it"):
        report = reports[language]
        assert report["language"] == language
        assert report["segments"]
        assert report["diarization"]["turns"]
        attributions = report["attributions"]
        assert len(attributions) == len(report["segments"])
        assert all(
            (segment["start_ms"], segment["end_ms"])
            == (attribution["start_ms"], attribution["end_ms"])
            for segment, attribution in zip(
                report["segments"], attributions, strict=True
            )
        )


def assert_complete_transcript(actual: str, expected: str) -> None:
    assert normalize_text(actual) == normalize_text(expected), (
        f"Expected: {expected!r}; actual: {actual!r}"
    )


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_offline_job_transcribes_all_two_voice_speech(
    attributed_job_report: str,
) -> None:
    report = json.loads(attributed_job_report)["en"]
    expected = (ROOT / "test/data/smoke/two-voices-script.txt").read_text(
        encoding="utf-8"
    )
    actual = " ".join(segment["text"] for segment in report["segments"])
    assert_complete_transcript(actual, expected)
