"""Verify transcript contract values are immutable and framework-independent."""

from dataclasses import FrozenInstanceError

import pytest

from app.asr_contract import ModelProvenance, Transcript, TranscriptSegment


def test_transcript_contract_is_immutable() -> None:
    provenance = ModelProvenance(
        "fake", "1.0", "model", "revision", "digest", "cpu", "int8", True, 5
    )
    segment = TranscriptSegment(0, 100, "No.")
    transcript = Transcript("en", 100, (segment,), provenance)
    with pytest.raises(FrozenInstanceError):
        segment.text = "Yes."  # type: ignore[misc]
    assert transcript.segments == (segment,)
    assert transcript.provenance == provenance
