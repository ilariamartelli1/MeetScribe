"""Test checkpoint reference parsing, scoring, and negation reporting."""

import json
from pathlib import Path

import pytest

from test.src.asr_evaluation import (
    evaluate,
    quality_failures,
    read_reference,
    scoring_text,
)


def identity(text: str) -> str:
    return text


def test_scoring_text_preserves_meaningful_symbols() -> None:
    assert scoring_text("Price: €2,400.50; change: -27!", identity) == (
        "price €2,400.50 change -27"
    )
    assert scoring_text("No, non.", identity) == "no non"


def test_wer_counts_edits_and_reports_negation() -> None:
    result = evaluate(
        "we do not release today", "we do release tomorrow", "en", identity
    )
    assert result.reference_words == 5
    assert result.deletions == 1
    assert result.substitutions == 1
    assert result.insertions == 0
    assert result.wer == pytest.approx(0.4)
    assert result.negation_differences
    assert "not" in result.alignment


@pytest.mark.parametrize(
    ("reference", "hypothesis", "language"),
    [
        ("non approviamo", "approviamo", "it"),
        ("we approve", "we do not approve", "en"),
        ("mai approvato", "approvato", "it"),
    ],
)
def test_missing_or_extra_negations_are_reported(
    reference: str, hypothesis: str, language: str
) -> None:
    assert evaluate(reference, hypothesis, language, identity).negation_differences


def test_unrelated_error_does_not_report_negation() -> None:
    result = evaluate("we do not approve", "we do not agree", "en", identity)
    assert result.substitutions == 1
    assert not result.negation_differences


def test_wer_threshold_is_inclusive() -> None:
    words = " ".join(f"word{i}" for i in range(100))
    exact_limit = (
        words.replace("word0 ", "changed ", 1)
        .replace("word1 ", "changed ", 1)
        .replace("word2 ", "changed ", 1)
    )
    over_limit = exact_limit.replace("word3 ", "changed ", 1)
    assert not quality_failures(evaluate(words, exact_limit, "en", identity), 0.03)
    assert quality_failures(evaluate(words, over_limit, "en", identity), 0.03)


def test_negation_difference_fails_even_under_threshold() -> None:
    reference = " ".join(["word"] * 100 + ["not", "approved"])
    hypothesis = " ".join(["word"] * 100 + ["approved"])
    result = evaluate(reference, hypothesis, "en", identity)
    assert result.wer < 0.03
    assert quality_failures(result, 0.03)


def test_invalid_wer_limit_is_rejected() -> None:
    result = evaluate("word", "word", "en", identity)
    with pytest.raises(ValueError, match="threshold"):
        quality_failures(result, 1.1)


def test_empty_or_unsupported_inputs_fail() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate("test", "", "en", identity)
    with pytest.raises(ValueError, match="Unsupported language"):
        evaluate("test", "test", "fr", identity)


def test_script_and_json_reference_must_agree(tmp_path: Path) -> None:
    script = tmp_path / "script.txt"
    reference = tmp_path / "reference.json"
    script.write_text("HEADER\n[1–2 ms] First sentence.\n[3–4 ms] Second sentence.\n")
    reference.write_text(
        json.dumps(
            {
                "segments": [
                    {"sentence": "First sentence."},
                    {"sentence": "Second sentence."},
                ]
            }
        )
    )
    assert read_reference(script, reference) == "First sentence. Second sentence."
    reference.write_text(json.dumps({"segments": [{"sentence": "Changed."}]}))
    with pytest.raises(ValueError, match="differ"):
        read_reference(script, reference)
