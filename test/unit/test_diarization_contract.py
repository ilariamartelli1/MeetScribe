"""Verify the model-independent diarization data contract."""

from dataclasses import replace

import pytest

from app.core.diarization_contract import Diarization, DiarizationProvenance


def test_result_is_immutable() -> None:
    provenance = DiarizationProvenance("fake", "1", "model", "revision", "cpu")
    result = Diarization(1000, (), (), provenance)
    assert replace(result, duration_ms=2000).duration_ms == 2000
    with pytest.raises(AttributeError):
        result.duration_ms = 2000  # type: ignore[misc]
