"""Parse draft minutes and validate their links to source transcript evidence."""

import re
from collections.abc import Sequence
from datetime import date

from pydantic import ValidationError

from app.minutes_contract import (
    MeetingExportV1,
    MinutesActionItem,
    MinutesDecision,
    MinutesDraft,
    MinutesTopic,
    ReviewStatus,
    TranscriptSegmentV1,
)
from app.speaker_alignment import AttributedTranscript


class MinutesResponseShapeError(ValueError):
    """The model response is not valid JSON matching the minutes contract."""


class MinutesCompletenessError(ValueError):
    """The model response omits required minutes content or citations."""


class MinutesEvidenceError(ValueError):
    """The minutes refer to unsupported transcript evidence."""


class InvalidTranscriptEvidenceError(ValueError):
    """The transcript and speaker attributions cannot form an export contract."""


def parse_minutes_response(response: str) -> MinutesDraft:
    try:
        return MinutesDraft.model_validate_json(response)
    except ValidationError as exc:
        raise MinutesResponseShapeError(
            "The summary model returned minutes that do not match the required schema."
        ) from exc


def validate_minutes_completeness(
    minutes: MinutesDraft, *, require_citations: bool = True
) -> None:
    if not minutes.summary.strip():
        raise MinutesCompletenessError("Minutes summary must not be empty.")

    collections: tuple[
        tuple[str, Sequence[MinutesTopic | MinutesDecision | MinutesActionItem]], ...
    ] = (
        ("topic", minutes.topics),
        ("decision", minutes.decisions),
        ("action item", minutes.action_items),
    )
    for kind, items in collections:
        for index, item in enumerate(items, start=1):
            text = (
                item.description if isinstance(item, MinutesActionItem) else item.text
            )
            if not text.strip():
                raise MinutesCompletenessError(
                    f"Minutes {kind} {index} must not be empty."
                )
            citations = item.source_segment_ids
            if (require_citations and not citations) or any(
                not citation.strip() for citation in citations
            ):
                raise MinutesCompletenessError(
                    f"Minutes {kind} {index} must cite at least one source segment."
                )
            if len(set(citations)) != len(citations):
                raise MinutesCompletenessError(
                    f"Minutes {kind} {index} contains duplicate source segment IDs."
                )

    for index, item in enumerate(minutes.action_items, start=1):
        for field_name, value in (
            ("owner_name", item.owner_name),
            ("owner_speaker_id", item.owner_speaker_id),
            ("deadline_text", item.deadline_text),
        ):
            if value is not None and not value.strip():
                raise MinutesCompletenessError(
                    f"Minutes action item {index} has an empty {field_name}; use null."
                )


def validate_minutes_citations(
    minutes: MinutesDraft, segments: Sequence[TranscriptSegmentV1]
) -> None:
    segment_by_id = {segment.id: segment for segment in segments}
    if len(segment_by_id) != len(segments):
        raise MinutesEvidenceError("Transcript segment IDs must be unique.")

    for index, topic in enumerate(minutes.topics, start=1):
        _validate_citation_ids("topic", index, topic.source_segment_ids, segment_by_id)
    for index, decision in enumerate(minutes.decisions, start=1):
        _validate_citation_ids(
            "decision", index, decision.source_segment_ids, segment_by_id
        )
    for index, action_item in enumerate(minutes.action_items, start=1):
        _validate_citation_ids(
            "action item",
            index,
            action_item.source_segment_ids,
            segment_by_id,
        )


def validate_minutes_evidence(
    minutes: MinutesDraft, segments: Sequence[TranscriptSegmentV1]
) -> None:
    validate_minutes_completeness(minutes)
    validate_minutes_dates(minutes)
    validate_minutes_citations(minutes, segments)
    segment_by_id = {segment.id: segment for segment in segments}
    available_speaker_ids = {
        segment.speaker_id for segment in segments if segment.speaker_id is not None
    }

    for index, item in enumerate(minutes.action_items, start=1):
        cited_segments = [
            segment_by_id[segment_id] for segment_id in item.source_segment_ids
        ]
        cited_text = " ".join(segment.text for segment in cited_segments)

        if item.owner_name is not None:
            owner_name = item.owner_name.strip()
            normalized_name = owner_name.casefold().strip(" \t.,:;!?")
            if normalized_name in {
                "i",
                "we",
                "me",
                "us",
                "myself",
                "ourselves",
                "io",
                "noi",
                "ci",
            }:
                raise MinutesEvidenceError(
                    f"Action item {index} must not use a pronoun as owner_name."
                )
            if re.fullmatch(r"voice\s+\d+", normalized_name):
                raise MinutesEvidenceError(
                    f"Action item {index} must keep anonymous voice IDs separate "
                    "from owner_name."
                )
            if normalized_name in {"unknown", "unspecified", "not specified"}:
                raise MinutesEvidenceError(
                    f"Action item {index} must use null for an unspecified owner."
                )
            if not _contains_name(cited_text, owner_name):
                raise MinutesEvidenceError(
                    f"Action item {index} owner_name is not supported by its cited "
                    "transcript segments."
                )

        if item.owner_speaker_id is not None:
            if item.owner_speaker_id not in available_speaker_ids:
                raise MinutesEvidenceError(
                    f"Action item {index} contains an unknown owner_speaker_id."
                )
            if not any(
                segment.speaker_id == item.owner_speaker_id
                for segment in cited_segments
            ):
                raise MinutesEvidenceError(
                    f"Action item {index} owner_speaker_id is not present in its "
                    "cited transcript segments."
                )

        if item.deadline_text is not None:
            if _normalize_text(item.deadline_text) not in _normalize_text(cited_text):
                raise MinutesEvidenceError(
                    f"Action item {index} deadline_text is not present in its cited "
                    "transcript segments."
                )


def validate_minutes_dates(minutes: MinutesDraft) -> None:
    for index, item in enumerate(minutes.action_items, start=1):
        if item.deadline_date is None:
            continue
        if item.deadline_text is None:
            raise MinutesEvidenceError(
                f"Action item {index} cannot have deadline_date without deadline_text."
            )
        explicit_date = _explicit_absolute_date(item.deadline_text)
        if explicit_date is None or explicit_date != item.deadline_date:
            raise MinutesEvidenceError(
                f"Action item {index} deadline_date is not an unambiguous "
                "absolute date in its deadline_text."
            )


def transcript_segments_v1(
    transcript: AttributedTranscript,
) -> tuple[TranscriptSegmentV1, ...]:
    if transcript.duration_ms <= 0 or not transcript.segments:
        raise InvalidTranscriptEvidenceError(
            "Transcript must have a positive duration and at least one segment."
        )
    if len(transcript.segments) != len(transcript.attributions):
        raise InvalidTranscriptEvidenceError(
            "Transcript segments and speaker attributions must have equal lengths."
        )

    speaker_ids: dict[str, str] = {}
    output: list[TranscriptSegmentV1] = []
    previous_start = -1
    previous_end = -1
    for index, (segment, attribution) in enumerate(
        zip(transcript.segments, transcript.attributions, strict=True), start=1
    ):
        if (
            not segment.text.strip()
            or not 0 <= segment.start_ms < segment.end_ms <= transcript.duration_ms
            or segment.start_ms < previous_start
            or segment.end_ms < previous_end
        ):
            raise InvalidTranscriptEvidenceError(
                "Transcript segments must contain text and ordered timestamps "
                "within the original recording."
            )
        if (
            attribution.start_ms != segment.start_ms
            or attribution.end_ms != segment.end_ms
        ):
            raise InvalidTranscriptEvidenceError(
                "Speaker attributions must use the matching transcript segment times."
            )

        speaker_id: str | None = None
        if attribution.speaker is not None:
            if not attribution.speaker.strip():
                raise InvalidTranscriptEvidenceError(
                    "Speaker attribution must be a non-empty label or null."
                )
            if attribution.speaker not in speaker_ids:
                speaker_ids[attribution.speaker] = f"voice-{len(speaker_ids) + 1}"
            speaker_id = speaker_ids[attribution.speaker]

        output.append(
            TranscriptSegmentV1(
                id=f"segment-{index}",
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                speaker_id=speaker_id,
                text=segment.text,
            )
        )
        previous_start = segment.start_ms
        previous_end = segment.end_ms
    return tuple(output)


def build_meeting_export_v1(
    meeting_id: str,
    transcript: AttributedTranscript,
    minutes: MinutesDraft,
    *,
    transcript_revision: int = 1,
    minutes_revision: int = 1,
    minutes_source_transcript_revision: int = 1,
    review_status: ReviewStatus = "DRAFT / UNREVIEWED",
    check_evidence: bool = True,
) -> MeetingExportV1:
    segments = transcript_segments_v1(transcript)
    if check_evidence:
        validate_minutes_evidence(minutes, segments)
    else:
        validate_minutes_completeness(minutes, require_citations=False)
        validate_minutes_citations(minutes, segments)
        validate_minutes_dates(minutes)
    return MeetingExportV1(
        schema_version=1,
        meeting_id=meeting_id,
        language=transcript.language,
        review_status=review_status,
        stale=minutes_source_transcript_revision != transcript_revision,
        transcript_revision=transcript_revision,
        minutes_revision=minutes_revision,
        minutes_source_transcript_revision=minutes_source_transcript_revision,
        segments=segments,
        minutes=minutes,
    )


def _contains_name(text: str, name: str) -> bool:
    normalized_text = text.casefold()
    normalized_name = name.casefold().strip()
    pattern = rf"(?<!\w){re.escape(normalized_name)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def _validate_citation_ids(
    kind: str,
    index: int,
    source_segment_ids: tuple[str, ...],
    segment_by_id: dict[str, TranscriptSegmentV1],
) -> None:
    unsupported = [
        segment_id
        for segment_id in source_segment_ids
        if segment_id not in segment_by_id
    ]
    if unsupported:
        raise MinutesEvidenceError(
            f"Minutes {kind} {index} cites unknown transcript segment "
            f"{unsupported[0]!r}."
        )


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "gennaio": 1,
    "gen": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "mag": 5,
    "giugno": 6,
    "giu": 6,
    "luglio": 7,
    "lug": 7,
    "agosto": 8,
    "ago": 8,
    "settembre": 9,
    "set": 9,
    "ottobre": 10,
    "ott": 10,
    "novembre": 11,
    "dicembre": 12,
    "dic": 12,
}
_MONTH_PATTERN = "|".join(
    re.escape(month) for month in sorted(_MONTHS, key=len, reverse=True)
)
_ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{4})(?P<separator>[-/.])"
    r"(?P<month>\d{1,2})(?P=separator)(?P<day>\d{1,2})(?!\d)"
)
_NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<first>\d{1,2})(?P<separator>[/.])"
    r"(?P<second>\d{1,2})(?P=separator)(?P<year>\d{4})(?!\d)"
)
_MONTH_MENTION_PATTERN = re.compile(
    rf"(?<!\w)(?:{_MONTH_PATTERN})\.?(?!\w)", re.IGNORECASE
)
_MONTH_FIRST_DATE_PATTERN = re.compile(
    rf"(?<!\w)(?P<month>{_MONTH_PATTERN})\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})(?!\w)",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_PATTERN = re.compile(
    rf"(?<!\w)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?:(?:of|di)\s+)?(?P<month>{_MONTH_PATTERN})\.?,?\s+"
    r"(?P<year>\d{4})(?!\w)",
    re.IGNORECASE,
)


def _explicit_absolute_date(text: str) -> date | None:
    candidates: set[date] = set()
    invalid_or_ambiguous = False
    found = False
    recognized_spans: list[tuple[int, int]] = []

    for match in _ISO_DATE_PATTERN.finditer(text):
        found = True
        recognized_spans.append(match.span())
        parsed = _make_date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        if parsed is None:
            invalid_or_ambiguous = True
        else:
            candidates.add(parsed)

    for match in _NUMERIC_DATE_PATTERN.finditer(text):
        found = True
        recognized_spans.append(match.span())
        parsed = _parse_unambiguous_numeric_date(
            int(match.group("first")),
            int(match.group("second")),
            int(match.group("year")),
        )
        if parsed is None:
            invalid_or_ambiguous = True
        else:
            candidates.add(parsed)

    for pattern in (_MONTH_FIRST_DATE_PATTERN, _DAY_FIRST_DATE_PATTERN):
        for match in pattern.finditer(text):
            found = True
            recognized_spans.append(match.span())
            day = int(match.group("day"))
            month = _MONTHS[match.group("month").casefold()]
            year = int(match.group("year"))
            parsed = _make_date(year, month, day)
            if parsed is None:
                invalid_or_ambiguous = True
            else:
                candidates.add(parsed)

    for mention in _MONTH_MENTION_PATTERN.finditer(text):
        if not any(
            start <= mention.start() and mention.end() <= end
            for start, end in recognized_spans
        ):
            invalid_or_ambiguous = True

    if not found or invalid_or_ambiguous or len(candidates) != 1:
        return None
    return next(iter(candidates))


def _parse_unambiguous_numeric_date(first: int, second: int, year: int) -> date | None:
    if first > 12 and 1 <= second <= 12:
        return _make_date(year, second, first)
    if second > 12 and 1 <= first <= 12:
        return _make_date(year, first, second)
    if first == second and 1 <= first <= 12:
        return _make_date(year, first, first)
    return None


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
