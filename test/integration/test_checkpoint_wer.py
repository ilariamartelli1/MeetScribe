"""Measure synthetic checkpoint WER using real offline GPU transcription."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest
from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore[import-untyped]

from test.src.asr_evaluation import (
    evaluate,
    quality_failures,
    read_reference,
    scoring_text,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "test/data/checkpoint"
MAX_WER = 0.03


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_checkpoint_wer_and_negations(asr_image: str, tmp_path: Path) -> None:
    model_dir = Path(
        os.environ.get(
            "MEETSCRIBE_MODEL_DIR",
            str(ROOT / "models/faster-whisper-large-v3-turbo"),
        )
    ).resolve(strict=True)
    references = {}
    for language, folder in (("en", "english"), ("it", "italian")):
        base = CHECKPOINT / folder / f"meeting-{language}"
        reference = base.with_name(f"{base.name}-draft-reference.json")
        references[language] = read_reference(
            base.with_name(f"{base.name}-script.txt"), reference
        )
        hashes = json.loads(reference.read_text(encoding="utf-8"))["files_sha256"]
        for extension in ("wav", "mp3"):
            path = base.with_suffix(f".{extension}")
            with path.open("rb") as audio:
                assert (
                    hashlib.file_digest(audio, "sha256").hexdigest()
                    == hashes[path.name]
                )

    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--mount",
            f"type=bind,src={model_dir},dst=/model,readonly",
            "--mount",
            f"type=bind,src={CHECKPOINT},dst=/audio,readonly",
            "--mount",
            f"type=bind,src={ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/app",
            "--entrypoint",
            "python",
            asr_image,
            "/checks/checkpoint_transcription_smoke.py",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, (
        f"Checkpoint transcription failed:\n{run.stderr}\n{run.stdout}"
    )
    reports = json.loads(run.stdout)
    assert len(reports) == 4
    assert {report["file"] for report in reports} == {
        "meeting-en.wav",
        "meeting-en.mp3",
        "meeting-it.wav",
        "meeting-it.mp3",
    }

    normalizers: dict[str, Callable[[str], str]] = {}
    for language in ("en", "it"):
        cache = tmp_path / language
        cache.mkdir()
        normalizers[language] = Normalizer(
            input_case="lower_cased", lang=language, cache_dir=str(cache)
        ).normalize
        normalizer = normalizers[language]
        number = "twenty seven" if language == "en" else "ventisette"
        assert scoring_text(number, normalizer) == scoring_text("27", normalizer)
        assert scoring_text("-27", normalizer) != scoring_text("27", normalizer)
        assert scoring_text("€27", normalizer) != scoring_text("$27", normalizer)

    failures = []
    for report in reports:
        transcript = report["transcript"]
        language = transcript["language"]
        assert report["file"].startswith(f"meeting-{language}.")
        actual = " ".join(segment["text"] for segment in transcript["segments"])
        evaluation = evaluate(
            references[language], actual, language, normalizers[language]
        )
        print(
            f"\n{report['file']}: WER={evaluation.wer:.2%}; "
            f"reference words={evaluation.reference_words}, "
            f"substitutions={evaluation.substitutions}, deletions={evaluation.deletions}, "
            f"insertions={evaluation.insertions}; "
            f"negation differences={len(evaluation.negation_differences)}"
        )
        print(evaluation.error_counts)
        problems = quality_failures(evaluation, MAX_WER)
        if problems:
            print(evaluation.alignment)
            failures.append(f"{report['file']}: {problems}")
    assert not failures, "\n".join(failures)
