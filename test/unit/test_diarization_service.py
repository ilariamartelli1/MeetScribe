"""Verify the model-independent diarization contract."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.diarization_contract import (
    Diarization,
    DiarizationProvenance,
    SpeakerTurn,
)
from app.core.diarization_service import DiarizationService, InvalidDiarizationError

PROVENANCE = DiarizationProvenance("fake", "1", "model", "revision", "cpu")


def test_regular_overlap_is_preserved_and_exclusive_turns_are_separate() -> None:
    result = Diarization(
        duration_ms=10_000,
        turns=(SpeakerTurn(1000, 5000, "A"), SpeakerTurn(4000, 8000, "B")),
        exclusive_turns=(SpeakerTurn(1000, 5000, "A"), SpeakerTurn(5000, 8000, "B")),
        provenance=PROVENANCE,
    )
    diarizer = Mock()
    diarizer.diarize.return_value = result

    assert DiarizationService(diarizer).diarize(Path("meeting.wav")) == result
    diarizer.diarize.assert_called_once_with(Path("meeting.wav"))


@pytest.mark.parametrize(
    "result",
    [
        Diarization(0, (), (), PROVENANCE),
        Diarization(1000, (SpeakerTurn(-1, 200, "A"),), (), PROVENANCE),
        Diarization(1000, (SpeakerTurn(100, 100, "A"),), (), PROVENANCE),
        Diarization(1000, (SpeakerTurn(100, 1001, "A"),), (), PROVENANCE),
        Diarization(1000, (SpeakerTurn(100, 200, " "),), (), PROVENANCE),
        Diarization(
            1000,
            (SpeakerTurn(500, 700, "A"), SpeakerTurn(100, 200, "B")),
            (),
            PROVENANCE,
        ),
        Diarization(
            1000,
            (),
            (SpeakerTurn(100, 500, "A"), SpeakerTurn(400, 700, "B")),
            PROVENANCE,
        ),
    ],
)
def test_invalid_turns_and_duration_are_rejected(result: Diarization) -> None:
    diarizer = Mock()
    diarizer.diarize.return_value = result
    with pytest.raises(InvalidDiarizationError):
        DiarizationService(diarizer).diarize(Path("meeting.wav"))


def test_silence_and_model_errors_are_not_hidden() -> None:
    diarizer = Mock()
    diarizer.diarize.return_value = Diarization(1000, (), (), PROVENANCE)
    service = DiarizationService(diarizer)
    assert service.diarize(Path("silence.wav")).turns == ()
    diarizer.diarize.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        service.diarize(Path("meeting.wav"))
