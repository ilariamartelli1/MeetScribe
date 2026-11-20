"""Verify synthetic audio and its scripts match the versioned source fixtures."""

import json
from pathlib import Path

from test.src.audio_fixtures import (
    ESPEAK_INTERVALS_MS,
    SMOKE,
    VOICE_INTERVALS_MS,
    write_repeated_sentence,
    write_two_voices,
)


def test_two_voice_audio_matches_its_sources(tmp_path: Path) -> None:
    generated = tmp_path / "two-voices.wav"
    assert tuple(write_two_voices(generated)) == VOICE_INTERVALS_MS
    assert generated.read_bytes() == (SMOKE / "two-voices.wav").read_bytes()


def test_repetition_audio_matches_its_source(tmp_path: Path) -> None:
    generated = tmp_path / "repeated-sentence.wav"
    write_repeated_sentence(generated)
    assert generated.read_bytes() == (SMOKE / "repeated-sentence.wav").read_bytes()


def test_composite_scripts_match_source_scripts() -> None:
    first = (SMOKE / "synthetic-meeting-script.txt").read_text(encoding="utf-8").strip()
    alternate = (
        (SMOKE / "alternate-flite-script.txt").read_text(encoding="utf-8").strip()
    )
    checkpoint = json.loads(
        (SMOKE.parent / "checkpoint/english/meeting-en-draft-reference.json").read_text(
            encoding="utf-8"
        )
    )
    assert (SMOKE / "repeated-sentence-script.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [first, first]
    assert (SMOKE / "two-voices-script.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        first,
        checkpoint["segments"][0]["sentence"],
        alternate,
        checkpoint["segments"][2]["sentence"],
    ]

    segments = checkpoint["segments"]
    for (start, end), index in zip(ESPEAK_INTERVALS_MS, (0, 2), strict=True):
        assert start <= segments[index]["start_ms"] < segments[index]["end_ms"] <= end
        if index:
            assert segments[index - 1]["end_ms"] < start
        assert end < segments[index + 1]["start_ms"]


def test_quiet_negation_scripts_match_generator_inputs() -> None:
    for language in ("en", "it"):
        source = (SMOKE / f"quiet-negations-{language}-source.txt").read_text(
            encoding="utf-8"
        )
        reference = (SMOKE / f"quiet-negations-{language}-script.txt").read_text(
            encoding="utf-8"
        )
        assert " ".join(source.splitlines()) == reference.strip()


def test_checkpoint_generation_inputs_and_reviewed_references() -> None:
    for language, folder in (("en", "english"), ("it", "italian")):
        base = SMOKE.parent / f"checkpoint/{folder}/meeting-{language}"
        source = (
            base.with_name(f"{base.name}-source.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        reference = json.loads(
            base.with_name(f"{base.name}-draft-reference.json").read_text(
                encoding="utf-8"
            )
        )
        if language == "en":
            assert "no:" in source[15]
            source[15] = source[15].replace("no:", "no, colon,")
        assert source == [segment["sentence"] for segment in reference["segments"]]
