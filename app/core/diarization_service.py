"""Validate model-independent speaker turns."""

from pathlib import Path

from app.core.diarization_contract import Diarization, Diarizer, SpeakerTurn


class InvalidDiarizationError(ValueError):
    pass


def _validate_turns(
    turns: tuple[SpeakerTurn, ...], duration_ms: int, *, exclusive: bool
) -> None:
    previous_start = -1
    previous_end = -1
    for turn in turns:
        if (
            not turn.speaker_id.strip()
            or not 0 <= turn.start_ms < turn.end_ms <= duration_ms
            or turn.start_ms < previous_start
            or (exclusive and turn.start_ms < previous_end)
        ):
            raise InvalidDiarizationError(
                "Speaker turns must have an ID and ordered, valid timestamps; "
                "exclusive turns must not overlap."
            )
        previous_start = turn.start_ms
        previous_end = turn.end_ms


class DiarizationService:
    def __init__(self, diarizer: Diarizer) -> None:
        self._diarizer = diarizer

    def diarize(self, audio_path: Path) -> Diarization:
        result = self._diarizer.diarize(audio_path)
        if result.duration_ms <= 0:
            raise InvalidDiarizationError("Recording duration must be positive.")
        _validate_turns(result.turns, result.duration_ms, exclusive=False)
        _validate_turns(result.exclusive_turns, result.duration_ms, exclusive=True)
        return result
