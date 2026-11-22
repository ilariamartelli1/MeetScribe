"""Verify temporary minutes generation, editing, review and export."""

import asyncio
from datetime import date

import pytest

from app.core.asr_contract import ModelProvenance, Transcript, TranscriptSegment
from app.core.diarization_contract import Diarization, DiarizationProvenance
from app.core.meeting_minutes import (
    MeetingMinutesCoordinator,
    MinutesEditValidationError,
    MinutesJobCapacityError,
    MinutesJobNotFoundError,
    MinutesJobNotReadyError,
    MinutesJobSnapshot,
    MinutesRegenerationConfirmationRequired,
    MinutesRevisionConflict,
    MinutesSourceUpdateConflict,
    render_minutes_text,
    render_transcript_text,
)
from app.core.minutes_contract import (
    MinutesActionItem,
    MinutesDraft,
    MinutesTopic,
)
from app.core.minutes_extractor import (
    MinutesExtractionResult,
)
from app.core.speaker_alignment import AttributedTranscript, SpeakerAttribution

PROVENANCE = ModelProvenance("fake", "1", "model", "revision", "digest", "cpu")
DIARIZATION_PROVENANCE = DiarizationProvenance(
    "fake", "1", "diarizer", "revision", "cpu"
)


def _transcript(
    text: str = "Elena will send the plan by March 5, 2027.",
) -> AttributedTranscript:
    segment = TranscriptSegment(100, 900, text)
    return AttributedTranscript(
        language="en",
        duration_ms=1000,
        segments=(segment,),
        provenance=PROVENANCE,
        diarization=Diarization(1000, (), (), DIARIZATION_PROVENANCE),
        attributions=(SpeakerAttribution(100, 900, "Voice 1"),),
    )


def _minutes() -> MinutesDraft:
    return MinutesDraft(
        summary="The team agreed to review the plan.",
        topics=(MinutesTopic(text="Planning", source_segment_ids=("segment-1",)),),
        decisions=(),
        action_items=(
            MinutesActionItem(
                description="Send the plan",
                owner_name="Elena",
                owner_speaker_id="voice-1",
                deadline_text="March 5, 2027",
                deadline_date=date(2027, 3, 5),
                source_segment_ids=("segment-1",),
            ),
        ),
    )


class FakeExtractor:
    def __init__(
        self,
        *,
        wait: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.wait = wait
        self.error = error
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.transcripts: list[AttributedTranscript] = []
        self.minutes = _minutes()

    async def extract(
        self, transcript: AttributedTranscript
    ) -> MinutesExtractionResult:
        self.transcripts.append(transcript)
        self.started.set()
        if self.wait:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.error is not None:
            raise self.error
        return MinutesExtractionResult(
            minutes=self.minutes,
            included_segment_ids=("segment-1",),
            context_size=4096,
            prompt_token_upper_bound=256,
            prompt_tokens=128,
            generated_tokens=64,
        )


async def _start_and_wait(
    coordinator: MeetingMinutesCoordinator,
    job_id: str = "job-1",
    transcript: AttributedTranscript | None = None,
) -> MinutesJobSnapshot:
    await coordinator.start(job_id, transcript or _transcript())
    for _ in range(20):
        snapshot = await coordinator.get(job_id)
        if snapshot.state != "running":
            return snapshot
        await asyncio.sleep(0)
    pytest.fail("Minutes extraction did not finish.")


async def _wait_for_regeneration(
    coordinator: MeetingMinutesCoordinator,
    job_id: str = "job-1",
) -> MinutesJobSnapshot:
    for _ in range(20):
        snapshot = await coordinator.get(job_id)
        if not snapshot.regenerating:
            return snapshot
        await asyncio.sleep(0)
    pytest.fail("Minutes regeneration did not finish.")


def test_generation_returns_draft_without_exposing_transcript() -> None:
    async def scenario() -> None:
        coordinator = MeetingMinutesCoordinator(FakeExtractor())
        try:
            started = await coordinator.start("job-1", _transcript())
            assert started.state == "running"
            snapshot = await _start_and_wait(coordinator)
            assert snapshot.state == "succeeded"
            assert snapshot.minutes == _minutes()
            assert snapshot.minutes_revision == 1
            assert snapshot.review_status == "DRAFT / UNREVIEWED"
            assert snapshot.extraction_metrics is not None
            assert snapshot.extraction_metrics.context_size == 4096
            assert snapshot.extraction_metrics.prompt_token_upper_bound == 256
            assert snapshot.extraction_metrics.prompt_tokens == 128
            assert snapshot.extraction_metrics.generated_tokens == 64
            assert "transcript" not in snapshot.__dataclass_fields__
            assert "Elena will send the plan by March 5, 2027." not in repr(snapshot)
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_extraction_failure_is_visible_and_cannot_be_exported() -> None:
    async def scenario() -> None:
        coordinator = MeetingMinutesCoordinator(
            FakeExtractor(error=RuntimeError("The local model is unavailable."))
        )
        try:
            snapshot = await _start_and_wait(coordinator)
            assert snapshot.state == "failed"
            assert snapshot.error is not None
            assert snapshot.error.kind == "RuntimeError"
            assert snapshot.error.message == "The local model is unavailable."
            with pytest.raises(MinutesJobNotReadyError):
                await coordinator.export("job-1")
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_cancellation_waits_for_extractor_to_stop() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor(wait=True)
        coordinator = MeetingMinutesCoordinator(extractor)
        try:
            await coordinator.start("job-1", _transcript())
            await extractor.started.wait()
            snapshot = await coordinator.cancel("job-1")
            assert snapshot.state == "cancelled"
            assert extractor.cancelled
            with pytest.raises(MinutesJobNotReadyError):
                await coordinator.export("job-1")
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_edits_increment_revision_and_review_is_revision_bound() -> None:
    async def scenario() -> None:
        coordinator = MeetingMinutesCoordinator(FakeExtractor())
        try:
            await _start_and_wait(coordinator)
            reviewed = await coordinator.mark_reviewed("job-1", 1)
            assert reviewed.review_status == "REVIEWED"

            edited = _minutes().model_copy(
                update={
                    "summary": "The reviewed plan needs an update.",
                    "topics": (),
                    "action_items": (
                        MinutesActionItem(
                            description="Ask Marta to review the plan",
                            owner_name="Marta",
                            owner_speaker_id=None,
                            deadline_text=None,
                            deadline_date=None,
                            source_segment_ids=(),
                        ),
                    ),
                }
            )
            updated = await coordinator.update("job-1", 1, edited)
            assert updated.minutes_revision == 2
            assert updated.review_status == "DRAFT / UNREVIEWED"
            assert updated.minutes == edited
            with pytest.raises(MinutesRevisionConflict):
                await coordinator.update("job-1", 1, edited)
            with pytest.raises(MinutesRevisionConflict):
                await coordinator.mark_reviewed("job-1", 1)

            exported = await coordinator.export("job-1")
            assert exported.minutes_revision == 2
            assert exported.review_status == "DRAFT / UNREVIEWED"
            assert exported.minutes.action_items[0].owner_name == "Marta"
            assert exported.minutes.action_items[0].source_segment_ids == ()
            text = render_minutes_text(exported)
            assert text.startswith("DRAFT / UNREVIEWED\n")
            assert "Marta" in text
            assert "Elena will send the plan by March 5, 2027." not in text

            reviewed = await coordinator.mark_reviewed("job-1", 2)
            assert reviewed.review_status == "REVIEWED"
            assert (await coordinator.export("job-1")).review_status == "REVIEWED"
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_invalid_manual_dates_and_citations_are_rejected() -> None:
    async def scenario() -> None:
        coordinator = MeetingMinutesCoordinator(FakeExtractor())
        try:
            await _start_and_wait(coordinator)
            invalid_date = _minutes().model_copy(
                update={
                    "action_items": (
                        MinutesActionItem(
                            description="Send the plan",
                            owner_name=None,
                            owner_speaker_id=None,
                            deadline_text=None,
                            deadline_date=date(2027, 3, 5),
                            source_segment_ids=(),
                        ),
                    )
                }
            )
            with pytest.raises(MinutesEditValidationError, match="without"):
                await coordinator.update("job-1", 1, invalid_date)

            invalid_citation = _minutes().model_copy(
                update={
                    "topics": (
                        MinutesTopic(text="Planning", source_segment_ids=("missing",)),
                    )
                }
            )
            with pytest.raises(MinutesEditValidationError, match="unknown transcript"):
                await coordinator.update("job-1", 1, invalid_citation)
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_retention_and_capacity_follow_transcription_results() -> None:
    async def scenario() -> None:
        now = [0.0]
        coordinator = MeetingMinutesCoordinator(
            FakeExtractor(),
            result_lifetime_seconds=10,
            max_results=1,
            clock=lambda: now[0],
        )
        try:
            await _start_and_wait(coordinator, "job-1")
            with pytest.raises(MinutesJobCapacityError):
                await coordinator.start("job-2", _transcript())
            await coordinator.retain({"job-2"})
            assert (await _start_and_wait(coordinator, "job-2")).state == "succeeded"
            now[0] = 10
            with pytest.raises(MinutesJobNotFoundError):
                await coordinator.get("job-2")
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_source_replacement_marks_minutes_stale_until_explicit_regeneration() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        original = _transcript()
        updated_source = _transcript(
            "Marta will send the updated plan by March 6, 2027."
        )
        try:
            await _start_and_wait(coordinator, transcript=original)
            reviewed = await coordinator.mark_reviewed("job-1", 1)
            assert reviewed.review_status == "REVIEWED"

            stale = await coordinator.start("job-1", updated_source)
            assert stale.state == "succeeded"
            assert stale.stale
            assert stale.transcript_revision == 2
            assert stale.minutes_source_transcript_revision == 1
            assert stale.review_status == "DRAFT / UNREVIEWED"
            assert extractor.transcripts == [original]
            with pytest.raises(MinutesJobNotReadyError, match="Stale minutes"):
                await coordinator.mark_reviewed("job-1", 1)

            stale_export = await coordinator.export("job-1")
            assert stale_export.stale
            assert stale_export.transcript_revision == 2
            assert stale_export.minutes_source_transcript_revision == 1
            assert stale_export.segments[0].text == original.segments[0].text
            assert "STALE:" in render_minutes_text(stale_export)

            extractor.minutes = _minutes().model_copy(
                update={
                    "summary": "Marta will send the updated plan.",
                    "action_items": (
                        MinutesActionItem(
                            description="Send the updated plan",
                            owner_name="Marta",
                            owner_speaker_id="voice-1",
                            deadline_text="March 6, 2027",
                            deadline_date=date(2027, 3, 6),
                            source_segment_ids=("segment-1",),
                        ),
                    ),
                }
            )
            started = await coordinator.regenerate("job-1", 1)
            assert started.regenerating
            regenerated = await _wait_for_regeneration(coordinator)
            assert regenerated.state == "succeeded"
            assert not regenerated.stale
            assert regenerated.minutes_revision == 2
            assert regenerated.minutes_source_transcript_revision == 2
            assert regenerated.review_status == "DRAFT / UNREVIEWED"
            assert extractor.transcripts == [original, updated_source]
            current_export = await coordinator.export("job-1")
            assert not current_export.stale
            assert current_export.segments[0].text == updated_source.segments[0].text
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_regeneration_failure_preserves_existing_minutes() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        try:
            original = await _start_and_wait(coordinator)
            extractor.error = RuntimeError("The replacement draft failed.")
            await coordinator.regenerate("job-1", 1)
            failed = await _wait_for_regeneration(coordinator)
            assert failed.state == "succeeded"
            assert not failed.regenerating
            assert failed.minutes == original.minutes
            assert failed.minutes_revision == 1
            assert failed.regeneration_error is not None
            assert failed.regeneration_error.message == "The replacement draft failed."
            assert (await coordinator.export("job-1")).minutes == original.minutes
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_regeneration_requires_confirmation_to_replace_manual_edits() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        try:
            await _start_and_wait(coordinator)
            manual_minutes = _minutes().model_copy(
                update={
                    "summary": "Human-edited summary.",
                    "topics": (),
                    "action_items": (),
                }
            )
            edited = await coordinator.update("job-1", 1, manual_minutes)
            with pytest.raises(MinutesRegenerationConfirmationRequired):
                await coordinator.regenerate("job-1", edited.minutes_revision or 0)
            assert (await coordinator.get("job-1")).minutes == manual_minutes

            await coordinator.regenerate(
                "job-1",
                edited.minutes_revision or 0,
                confirm_overwrite=True,
            )
            regenerated = await _wait_for_regeneration(coordinator)
            assert regenerated.minutes == _minutes()
            assert regenerated.minutes_revision == 3
            assert regenerated.review_status == "DRAFT / UNREVIEWED"
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_concurrent_edit_cannot_overwrite_regeneration() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        try:
            await _start_and_wait(coordinator)
            extractor.wait = True
            extractor.started = asyncio.Event()
            extractor.release = asyncio.Event()
            await coordinator.regenerate("job-1", 1)
            await extractor.started.wait()

            concurrent_edit = _minutes().model_copy(
                update={"summary": "Concurrent edit."}
            )
            with pytest.raises(MinutesJobNotReadyError):
                await coordinator.update("job-1", 1, concurrent_edit)
            extractor.release.set()
            regenerated = await _wait_for_regeneration(coordinator)
            assert regenerated.minutes_revision == 2
            with pytest.raises(MinutesRevisionConflict):
                await coordinator.update("job-1", 1, concurrent_edit)
            assert regenerated.minutes == _minutes()
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_cancelled_regeneration_keeps_previous_minutes() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        try:
            original = await _start_and_wait(coordinator)
            extractor.wait = True
            extractor.started = asyncio.Event()
            extractor.release = asyncio.Event()
            await coordinator.regenerate("job-1", 1)
            await extractor.started.wait()
            cancelled = await coordinator.cancel("job-1")
            assert cancelled.state == "succeeded"
            assert not cancelled.regenerating
            assert cancelled.regeneration_cancelled
            assert cancelled.minutes == original.minutes
            assert cancelled.minutes_revision == 1
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_source_replacement_waits_for_active_regeneration() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor()
        coordinator = MeetingMinutesCoordinator(extractor)
        original = _transcript()
        updated_source = _transcript(
            "Marta will send the updated plan by March 6, 2027."
        )
        try:
            await _start_and_wait(coordinator, transcript=original)
            extractor.wait = True
            extractor.started = asyncio.Event()
            extractor.release = asyncio.Event()
            await coordinator.regenerate("job-1", 1)
            await extractor.started.wait()

            with pytest.raises(MinutesSourceUpdateConflict):
                await coordinator.start("job-1", updated_source)
            extractor.release.set()
            await _wait_for_regeneration(coordinator)
            stale = await coordinator.start("job-1", updated_source)
            assert stale.stale
            assert stale.transcript_revision == 2
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_close_cancels_inference_and_discards_temporary_state() -> None:
    async def scenario() -> None:
        extractor = FakeExtractor(wait=True)
        coordinator = MeetingMinutesCoordinator(extractor)
        await coordinator.start("job-1", _transcript())
        await extractor.started.wait()
        await coordinator.close()
        assert extractor.cancelled
        with pytest.raises(MinutesJobNotFoundError):
            await coordinator.get("job-1")

    asyncio.run(scenario())


def test_transcript_download_text_is_separate_and_keeps_speaker_warning() -> None:
    transcript = _transcript()
    text = render_transcript_text(transcript)

    assert text.startswith("DRAFT / UNREVIEWED\n")
    assert "Speaker labels are anonymous and may be incorrect." in text
    assert "[00:00:00.100 - 00:00:00.900] Voice 1:" in text
    assert transcript.segments[0].text in text

    plain = Transcript(
        language="it",
        duration_ms=1000,
        segments=(TranscriptSegment(100, 900, "Città"),),
        provenance=PROVENANCE,
    )
    plain_text = render_transcript_text(plain)
    assert "[00:00:00.100 - 00:00:00.900] Città" in plain_text
    assert "Voice" not in plain_text
