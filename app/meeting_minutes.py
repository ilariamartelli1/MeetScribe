"""Temporary draft-minutes workflow, editing, review and export."""

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from app.asr_contract import Transcript
from app.minutes_contract import MeetingExportV1, MinutesDraft, ReviewStatus
from app.minutes_extractor import (
    MinutesExtractionError,
    MinutesExtractionResult,
    MinutesExtractor,
)
from app.minutes_validation import (
    InvalidTranscriptEvidenceError,
    MinutesCompletenessError,
    MinutesEvidenceError,
    build_meeting_export_v1,
    transcript_segments_v1,
    validate_minutes_completeness,
    validate_minutes_dates,
)
from app.speaker_alignment import AttributedTranscript

MinutesJobState = Literal["running", "cancelling", "cancelled", "succeeded", "failed"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinutesJobError:
    kind: str
    message: str


@dataclass(frozen=True)
class MinutesExtractionMetrics:
    context_size: int
    prompt_token_upper_bound: int
    prompt_tokens: int
    generated_tokens: int


@dataclass(frozen=True)
class MinutesJobSnapshot:
    job_id: str
    state: MinutesJobState
    minutes: MinutesDraft | None = None
    error: MinutesJobError | None = None
    transcript_revision: int = 1
    minutes_revision: int | None = None
    minutes_source_transcript_revision: int | None = None
    review_status: ReviewStatus | None = None
    stale: bool = False
    regenerating: bool = False
    regeneration_cancelling: bool = False
    regeneration_error: MinutesJobError | None = None
    regeneration_cancelled: bool = False
    extraction_metrics: MinutesExtractionMetrics | None = None


@dataclass
class _Record:
    transcript: AttributedTranscript
    state: MinutesJobState
    task: asyncio.Task[None] | None = None
    minutes: MinutesDraft | None = None
    minutes_source_transcript: AttributedTranscript | None = None
    extraction_metrics: MinutesExtractionMetrics | None = None
    error: MinutesJobError | None = None
    transcript_revision: int = 1
    minutes_revision: int | None = None
    minutes_source_transcript_revision: int | None = None
    reviewed_revision: int | None = None
    evidence_checked: bool = True
    regenerating: bool = False
    regeneration_cancelling: bool = False
    regeneration_error: MinutesJobError | None = None
    regeneration_cancelled: bool = False
    finished_at: float | None = None


class MinutesJobNotFoundError(LookupError):
    pass


class MinutesJobNotReadyError(RuntimeError):
    pass


class MinutesJobFinishedError(RuntimeError):
    pass


class MinutesJobCapacityError(RuntimeError):
    pass


class MinutesRevisionConflict(RuntimeError):
    pass


class MinutesRegenerationConfirmationRequired(RuntimeError):
    pass


class MinutesSourceUpdateConflict(RuntimeError):
    pass


class MinutesEditValidationError(ValueError):
    pass


class MinutesCoordinatorClosedError(RuntimeError):
    pass


class MeetingMinutesCoordinator:
    """Own transient extraction tasks and the editable state for each job."""

    def __init__(
        self,
        extractor: MinutesExtractor,
        *,
        result_lifetime_seconds: float = 3600,
        max_results: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not math.isfinite(result_lifetime_seconds)
            or result_lifetime_seconds <= 0
            or type(max_results) is not int
            or max_results < 1
        ):
            raise ValueError("Minutes result lifetime and capacity must be positive.")
        self._extractor = extractor
        self._lifetime = result_lifetime_seconds
        self._capacity = max_results
        self._clock = clock
        self._lock = asyncio.Lock()
        self._records: dict[str, _Record] = {}
        self._closed = False

    async def start(
        self, job_id: str, transcript: AttributedTranscript
    ) -> MinutesJobSnapshot:
        if not job_id.strip():
            raise ValueError("A job ID is required.")
        async with self._lock:
            self._ensure_open()
            self._expire_results()
            existing = self._records.get(job_id)
            if existing is not None:
                source_changed = existing.transcript != transcript
                if existing.state in ("running", "cancelling") or existing.regenerating:
                    if source_changed:
                        raise MinutesSourceUpdateConflict(
                            "The transcript cannot change while minutes are "
                            "being generated."
                        )
                    return self._snapshot(job_id, existing)
                if existing.state == "succeeded":
                    if source_changed:
                        existing.transcript = transcript
                        existing.transcript_revision += 1
                        existing.reviewed_revision = None
                        existing.regeneration_error = None
                        existing.regeneration_cancelled = False
                    return self._snapshot(job_id, existing)
                transcript_revision = existing.transcript_revision + int(source_changed)
                del self._records[job_id]
            else:
                transcript_revision = 1
            if len(self._records) >= self._capacity:
                raise MinutesJobCapacityError(
                    "Minutes result capacity reached; replace or wait for an older "
                    "temporary result to expire."
                )
            record = _Record(
                transcript=transcript,
                state="running",
                transcript_revision=transcript_revision,
            )
            self._records[job_id] = record
            self._start_extraction(job_id, record, is_regeneration=False)
            return self._snapshot(job_id, record)

    async def get(self, job_id: str) -> MinutesJobSnapshot:
        async with self._lock:
            self._expire_results()
            record = self._records.get(job_id)
            if record is None:
                raise MinutesJobNotFoundError(
                    f"Minutes for job {job_id} are not available."
                )
            return self._snapshot(job_id, record)

    async def cancel(self, job_id: str) -> MinutesJobSnapshot:
        async with self._lock:
            self._ensure_open()
            self._expire_results()
            record = self._records.get(job_id)
            if record is None:
                raise MinutesJobNotFoundError(
                    f"Minutes for job {job_id} are not available."
                )
            if record.regenerating:
                record.regeneration_cancelling = True
                task = record.task
            elif record.state == "cancelled":
                return self._snapshot(job_id, record)
            elif record.state in ("succeeded", "failed"):
                raise MinutesJobFinishedError(
                    f"Minutes generation for job {job_id} has already finished."
                )
            else:
                record.state = "cancelling"
                task = record.task
            if task is None:
                raise RuntimeError("Minutes generation task was not started.")
            if not task.done():
                task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        async with self._lock:
            current = self._records.get(job_id)
            if current is None:
                raise MinutesJobNotFoundError(
                    f"Minutes for job {job_id} are not available."
                )
            if current.state == "cancelling":
                current.state = "cancelled"
                current.finished_at = self._clock()
            return self._snapshot(job_id, current)

    async def regenerate(
        self,
        job_id: str,
        expected_revision: int,
        *,
        confirm_overwrite: bool = False,
    ) -> MinutesJobSnapshot:
        async with self._lock:
            self._ensure_open()
            self._expire_results()
            record = self._records.get(job_id)
            if record is None:
                raise MinutesJobNotFoundError(
                    f"Minutes for job {job_id} are not available."
                )
            if record.state != "succeeded" or record.minutes is None:
                raise MinutesJobNotReadyError(
                    f"Minutes for job {job_id} are not ready."
                )
            self._check_revision(record, expected_revision)
            if record.regenerating:
                raise MinutesJobNotReadyError(
                    f"Minutes for job {job_id} are already being regenerated."
                )
            if not record.evidence_checked and not confirm_overwrite:
                raise MinutesRegenerationConfirmationRequired(
                    "Regeneration would replace manually edited minutes; "
                    "explicit confirmation is required."
                )
            record.regenerating = True
            record.regeneration_cancelling = False
            record.regeneration_error = None
            record.regeneration_cancelled = False
            record.finished_at = None
            self._start_extraction(job_id, record, is_regeneration=True)
            return self._snapshot(job_id, record)

    async def update(
        self,
        job_id: str,
        expected_revision: int,
        minutes: MinutesDraft,
    ) -> MinutesJobSnapshot:
        async with self._lock:
            record = self._ready_record(job_id)
            self._check_revision(record, expected_revision)
            if minutes == record.minutes:
                return self._snapshot(job_id, record)
            next_revision = (record.minutes_revision or 0) + 1
            assert record.minutes_source_transcript is not None
            try:
                validate_minutes_completeness(minutes, require_citations=False)
                validate_minutes_dates(minutes)
                build_meeting_export_v1(
                    job_id,
                    record.minutes_source_transcript,
                    minutes,
                    transcript_revision=record.transcript_revision,
                    minutes_revision=next_revision,
                    minutes_source_transcript_revision=(
                        record.minutes_source_transcript_revision or 1
                    ),
                    check_evidence=False,
                )
            except (
                MinutesCompletenessError,
                MinutesEvidenceError,
                InvalidTranscriptEvidenceError,
                ValidationError,
            ) as exc:
                raise MinutesEditValidationError(str(exc)) from exc
            record.minutes = minutes
            record.minutes_revision = next_revision
            record.reviewed_revision = None
            record.evidence_checked = False
            return self._snapshot(job_id, record)

    async def mark_reviewed(
        self, job_id: str, expected_revision: int
    ) -> MinutesJobSnapshot:
        async with self._lock:
            record = self._ready_record(job_id)
            self._check_revision(record, expected_revision)
            if record.minutes_revision is None:
                raise MinutesJobNotReadyError(
                    f"Minutes for job {job_id} are not ready."
                )
            if self._snapshot(job_id, record).stale:
                raise MinutesJobNotReadyError(
                    "Stale minutes cannot be reviewed; regenerate them from the "
                    "current transcript first."
                )
            record.reviewed_revision = record.minutes_revision
            return self._snapshot(job_id, record)

    async def export(self, job_id: str) -> MeetingExportV1:
        async with self._lock:
            record = self._ready_record(job_id)
            assert record.minutes is not None
            assert record.minutes_source_transcript is not None
            assert record.minutes_revision is not None
            assert record.minutes_source_transcript_revision is not None
            snapshot = self._snapshot(job_id, record)
            assert snapshot.review_status is not None
            return build_meeting_export_v1(
                job_id,
                record.minutes_source_transcript,
                record.minutes,
                transcript_revision=record.transcript_revision,
                minutes_revision=record.minutes_revision,
                minutes_source_transcript_revision=(
                    record.minutes_source_transcript_revision
                ),
                review_status=snapshot.review_status,
                check_evidence=record.evidence_checked,
            )

    async def retain(self, retained_job_ids: Iterable[str]) -> None:
        retained = set(retained_job_ids)
        async with self._lock:
            self._expire_results()
            expired = [
                job_id
                for job_id, record in self._records.items()
                if job_id not in retained
                and record.state not in ("running", "cancelling")
                and not record.regenerating
            ]
            for job_id in expired:
                del self._records[job_id]

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = [
                record.task
                for record in self._records.values()
                if record.task is not None and not record.task.done()
            ]
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self._records.clear()

    def _start_extraction(
        self, job_id: str, record: _Record, *, is_regeneration: bool
    ) -> None:
        transcript = record.transcript
        transcript_revision = record.transcript_revision
        task_name = (
            f"minutes-regeneration-{job_id}" if is_regeneration else f"minutes-{job_id}"
        )
        record.task = asyncio.create_task(
            self._run(
                job_id,
                record,
                transcript,
                transcript_revision,
                is_regeneration=is_regeneration,
            ),
            name=task_name,
        )

    async def _run(
        self,
        job_id: str,
        record: _Record,
        transcript: AttributedTranscript,
        transcript_revision: int,
        *,
        is_regeneration: bool,
    ) -> None:
        try:
            result = await self._extractor.extract(transcript)
            if not isinstance(result, MinutesExtractionResult):
                raise MinutesExtractionError(
                    "The minutes extractor returned an invalid result."
                )
            expected_segment_ids = tuple(
                segment.id for segment in transcript_segments_v1(transcript)
            )
            if result.included_segment_ids != expected_segment_ids:
                raise MinutesExtractionError(
                    "The minutes extractor did not include every transcript segment."
                )
        except asyncio.CancelledError:
            async with self._lock:
                current = self._records.get(job_id)
                if current is record:
                    if is_regeneration:
                        current.regenerating = False
                        current.regeneration_cancelling = False
                        current.regeneration_error = None
                        current.regeneration_cancelled = True
                    else:
                        current.state = "cancelled"
                    current.finished_at = self._clock()
                    current.task = None
            raise
        except Exception as exc:
            logger.error(
                "Minutes extraction failed for job %s (%s).",
                job_id,
                type(exc).__name__,
            )
            async with self._lock:
                current = self._records.get(job_id)
                if current is record:
                    error = MinutesJobError(
                        type(exc).__name__, str(exc) or "Minutes extraction failed."
                    )
                    if is_regeneration:
                        current.regenerating = False
                        current.regeneration_cancelling = False
                        current.regeneration_error = error
                        current.regeneration_cancelled = False
                    else:
                        current.state = "failed"
                        current.error = error
                    current.finished_at = self._clock()
                    current.task = None
        else:
            async with self._lock:
                current = self._records.get(job_id)
                if current is not record:
                    return
                if current.transcript_revision != transcript_revision:
                    error = MinutesJobError(
                        "StaleTranscript",
                        "The transcript changed during generation; regenerate "
                        "minutes from the current revision.",
                    )
                    if is_regeneration:
                        current.regenerating = False
                        current.regeneration_cancelling = False
                        current.regeneration_error = error
                        current.regeneration_cancelled = False
                    else:
                        current.state = "failed"
                        current.error = error
                elif is_regeneration:
                    if current.regeneration_cancelling:
                        current.regeneration_cancelled = True
                    else:
                        assert current.minutes_revision is not None
                        current.minutes = result.minutes
                        current.minutes_source_transcript = transcript
                        current.minutes_revision += 1
                        current.minutes_source_transcript_revision = transcript_revision
                        current.extraction_metrics = MinutesExtractionMetrics(
                            result.context_size,
                            result.prompt_token_upper_bound,
                            result.prompt_tokens,
                            result.generated_tokens,
                        )
                        current.reviewed_revision = None
                        current.evidence_checked = True
                        current.regeneration_error = None
                        current.regeneration_cancelled = False
                    current.regenerating = False
                    current.regeneration_cancelling = False
                elif current.state == "cancelling":
                    current.state = "cancelled"
                else:
                    current.state = "succeeded"
                    current.minutes = result.minutes
                    current.minutes_revision = 1
                    current.minutes_source_transcript_revision = transcript_revision
                    current.minutes_source_transcript = transcript
                    current.extraction_metrics = MinutesExtractionMetrics(
                        result.context_size,
                        result.prompt_token_upper_bound,
                        result.prompt_tokens,
                        result.generated_tokens,
                    )
                    current.evidence_checked = True
                current.finished_at = self._clock()
                current.task = None

    def _ready_record(self, job_id: str) -> _Record:
        self._expire_results()
        record = self._records.get(job_id)
        if record is None:
            raise MinutesJobNotFoundError(
                f"Minutes for job {job_id} are not available."
            )
        if record.state != "succeeded" or record.minutes is None or record.regenerating:
            raise MinutesJobNotReadyError(f"Minutes for job {job_id} are not ready.")
        return record

    @staticmethod
    def _check_revision(record: _Record, expected_revision: int) -> None:
        if record.minutes_revision != expected_revision:
            raise MinutesRevisionConflict(
                "The minutes changed since they were loaded; reload before saving."
            )

    def _snapshot(self, job_id: str, record: _Record) -> MinutesJobSnapshot:
        review_status: ReviewStatus | None = None
        if record.minutes_revision is not None:
            review_status = (
                "REVIEWED"
                if record.reviewed_revision == record.minutes_revision
                else "DRAFT / UNREVIEWED"
            )
        stale = (
            record.minutes_source_transcript_revision is not None
            and record.minutes_source_transcript_revision != record.transcript_revision
        )
        return MinutesJobSnapshot(
            job_id=job_id,
            state=record.state,
            minutes=record.minutes,
            error=record.error,
            transcript_revision=record.transcript_revision,
            minutes_revision=record.minutes_revision,
            minutes_source_transcript_revision=(
                record.minutes_source_transcript_revision
            ),
            review_status=review_status,
            stale=stale,
            regenerating=record.regenerating,
            regeneration_cancelling=record.regeneration_cancelling,
            regeneration_error=record.regeneration_error,
            regeneration_cancelled=record.regeneration_cancelled,
            extraction_metrics=record.extraction_metrics,
        )

    def _expire_results(self) -> None:
        now = self._clock()
        expired = [
            job_id
            for job_id, record in self._records.items()
            if record.finished_at is not None
            and now - record.finished_at >= self._lifetime
        ]
        for job_id in expired:
            del self._records[job_id]

    def _ensure_open(self) -> None:
        if self._closed:
            raise MinutesCoordinatorClosedError("The minutes coordinator is closed.")


def render_minutes_text(export: MeetingExportV1) -> str:
    lines = [export.review_status, ""]
    if export.stale:
        lines.extend(
            (
                "STALE: these minutes cite transcript revision "
                f"{export.minutes_source_transcript_revision}; current revision is "
                f"{export.transcript_revision}.",
                "",
            )
        )
    lines.extend(("Summary", export.minutes.summary))
    for heading, items in (
        ("Topics", export.minutes.topics),
        ("Decisions", export.minutes.decisions),
    ):
        lines.extend(("", heading))
        lines.extend(f"- {item.text}" for item in items)
    lines.extend(("", "Action items"))
    for item in export.minutes.action_items:
        owner = item.owner_name or "Not specified"
        deadline = item.deadline_text or "Not specified"
        lines.append(f"- {item.description} | Owner: {owner} | Deadline: {deadline}")
    return "\n".join(lines) + "\n"


def render_transcript_text(transcript: Transcript | AttributedTranscript) -> str:
    lines = ["DRAFT / UNREVIEWED"]
    attributed_transcript = (
        transcript if isinstance(transcript, AttributedTranscript) else None
    )
    attributions = (
        attributed_transcript.attributions if attributed_transcript is not None else ()
    )
    if attributed_transcript is not None:
        if len(attributions) != len(transcript.segments):
            raise InvalidTranscriptEvidenceError(
                "Transcript segments and speaker attributions must have equal lengths."
            )
        if any(
            attribution.start_ms != segment.start_ms
            or attribution.end_ms != segment.end_ms
            for segment, attribution in zip(
                transcript.segments, attributions, strict=True
            )
        ):
            raise InvalidTranscriptEvidenceError(
                "Speaker attributions must use the matching transcript segment times."
            )
        lines.extend(
            (
                'Speaker labels are anonymous and may be incorrect. "Unknown speaker" '
                "means no unique speaker was assigned.",
                "",
            )
        )
    else:
        lines.append("")

    for index, segment in enumerate(transcript.segments):
        start = _format_timecode(segment.start_ms)
        end = _format_timecode(segment.end_ms)
        prefix = f"[{start} - {end}]"
        if attributed_transcript is not None:
            speaker = attributions[index].speaker or "Unknown speaker"
            prefix += f" {speaker}:"
        lines.extend((f"{prefix} {segment.text}", ""))
    return "\n".join(lines)


def _format_timecode(milliseconds: int) -> str:
    total = milliseconds
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, fraction = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{fraction:03}"
