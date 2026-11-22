"""Verify transcript contract values are immutable and framework-independent."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from app.core.asr_contract import (
    CPU_DEVICE,
    CUDA_DEVICE,
    ModelProvenance,
    Transcript,
    TranscriptSegment,
)


def test_device_values_match_external_interfaces() -> None:
    assert CPU_DEVICE == "cpu"
    assert CUDA_DEVICE == "cuda"


def test_transcript_contract_is_immutable() -> None:
    provenance = ModelProvenance("fake", "1.0", "model", "revision", "digest", "cpu")
    segment = TranscriptSegment(0, 100, "No.")
    transcript = Transcript("en", 100, (segment,), provenance)
    with pytest.raises(FrozenInstanceError):
        segment.text = "Yes."  # type: ignore[misc]
    assert transcript.segments == (segment,)
    assert transcript.provenance == provenance


def test_provider_can_supply_its_own_provenance_fields() -> None:
    @dataclass(frozen=True, kw_only=True)
    class OtherProvenance(ModelProvenance):
        language_detector: str

    provenance = OtherProvenance(
        adapter="other",
        adapter_version="1",
        model_id="other-model",
        model_revision="revision",
        language_detector="automatic",
    )
    transcript = Transcript(
        "en", 100, (TranscriptSegment(0, 100, "Hello"),), provenance
    )
    assert transcript.provenance is provenance
    assert provenance.language_detector == "automatic"
    assert not hasattr(provenance, "beam_size")
