"""Align speaker turns with transcript segments on the original timeline."""

from dataclasses import dataclass

from app.core.asr_contract import (
    Language,
    ModelProvenance,
    Transcript,
    TranscriptSegment,
)
from app.core.diarization_contract import Diarization, SpeakerTurn

MAX_DURATION_DIFFERENCE_MS = 20


@dataclass(frozen=True)
class SpeakerAttribution:
    start_ms: int
    end_ms: int
    speaker: str | None


@dataclass(frozen=True)
class AttributedTranscript:
    language: Language
    duration_ms: int
    segments: tuple[TranscriptSegment, ...]
    provenance: ModelProvenance
    diarization: Diarization
    attributions: tuple[SpeakerAttribution, ...]


def align_speakers(
    transcript: Transcript, diarization: Diarization
) -> AttributedTranscript:
    if (
        abs(transcript.duration_ms - diarization.duration_ms)
        > MAX_DURATION_DIFFERENCE_MS
    ):
        raise ValueError("Transcription and diarization audio durations differ.")

    labels: dict[str, str] = {}
    for turn in sorted(
        diarization.turns, key=lambda turn: (turn.start_ms, turn.speaker_id)
    ):
        if turn.speaker_id not in labels:
            labels[turn.speaker_id] = f"Voice {len(labels) + 1}"

    def overlaps(segment: TranscriptSegment, turn: SpeakerTurn) -> bool:
        return turn.start_ms < segment.end_ms and segment.start_ms < turn.end_ms

    attributions = []
    for segment in transcript.segments:
        speakers = {
            turn.speaker_id for turn in diarization.turns if overlaps(segment, turn)
        }
        speaker = labels[next(iter(speakers))] if len(speakers) == 1 else None
        attributions.append(
            SpeakerAttribution(segment.start_ms, segment.end_ms, speaker)
        )

    return AttributedTranscript(
        language=transcript.language,
        duration_ms=transcript.duration_ms,
        segments=transcript.segments,
        provenance=transcript.provenance,
        diarization=diarization,
        attributions=tuple(attributions),
    )
