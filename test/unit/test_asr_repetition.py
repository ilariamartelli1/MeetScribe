"""Verify the repeated-sentence diagnostic distinguishes expected outcomes."""

import pytest

from test.integration.test_asr_repetition import repetition_outcome

SENTENCE = "This is a test meeting. The project is on schedule."
SCRIPT = f"{SENTENCE}\n{SENTENCE}\n"


def test_complete_repetition_passes() -> None:
    assert repetition_outcome(f"{SENTENCE} {SENTENCE}", SCRIPT) == "complete"


def test_single_sentence_is_known_omission() -> None:
    assert repetition_outcome(SENTENCE, SCRIPT) == "omitted"


@pytest.mark.parametrize("actual", ["", "This is a different sentence.", SENTENCE * 3])
def test_unexpected_output_fails(actual: str) -> None:
    with pytest.raises(AssertionError, match="Unexpected repetition transcript"):
        repetition_outcome(actual, SCRIPT)


def test_invalid_script_is_rejected() -> None:
    with pytest.raises(ValueError, match="same sentence twice"):
        repetition_outcome(SENTENCE, SENTENCE)
