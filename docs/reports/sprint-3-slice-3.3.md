# Sprint 3.3 — Minutes schema and evidence checks

## Outcome

Added typed draft-minutes models, a strict JSON response parser, source-evidence
checks and the versioned export-v1 data contract. This slice defines and
validates the data passed between a future summary adapter and the API; it does
not generate minutes or change browser behavior.

## Contracts

`app/minutes_contract.py` defines immutable Pydantic models for topics,
decisions, action items, the model's `MinutesDraft` response, transcript
segments and `MeetingExportV1`. Unknown fields are rejected, every required key
must be present, and nullable owner/deadline fields must be explicitly present
as either a value or `null`. Empty top-level arrays are allowed. Pydantic is
now an explicit, version-pinned runtime dependency; FastAPI already depended
on the same locked version.

The export model follows the JSON v1 shape in `plan.md`. Its initial builder
uses the temporary job ID, language, original millisecond timestamps and
speaker IDs `voice-1`, `voice-2`, etc. Segment IDs are assigned in transcript
order and are stable for the same transcript revision. The builder owns the
initial `DRAFT / UNREVIEWED` status and revision values; the model response
cannot supply them. Persistent revision tracking and reviewed status remain
future work.

**Later cross-segment extraction behavior (slice 3.4):** The full export-v1
example in [plan.md](../../plan.md) has one segment asking about a new document
draft and another saying, "I'll send it next Friday." With both segments in
the prompt, the intended action description is `Send the new document draft`,
not just `Send it`, and it cites both segment IDs. This is contextual model
output, not behavior guaranteed by the JSON schema: validation checks citation
IDs and owner/deadline evidence, not whether the description resolves the
pronoun correctly. Human review remains necessary.

**Later stale-status behavior (slice 3.6):** `stale` marks a revision mismatch;
it does not indicate that the JSON file itself is old. It protects exports
when minutes cite an earlier transcript and a replacement transcript may reuse
segment IDs: the export keeps the cited source segments and marks the minutes
stale. The current upload flow assigns each audio a new job ID and provides no
UI/API action to re-transcribe or replace a job's transcript. So normal exports
contain `stale: false`; slice 3.6 adds handling for the mismatch and explicit
summary regeneration, not audio re-transcription. See the
[slice 3.6 report](sprint-3-slice-3.6.md).

## Validation boundaries

`app/minutes_validation.py` separates response-shape parsing,
`validate_minutes_completeness`, citation-ID checks and evidence checks. The
checks reject unknown segment IDs; require each listed topic, decision and
action item to cite source segments; keep a named owner separate from an
anonymous speaker ID; and require owner names and deadline wording to occur in
the cited text. Pronouns such as `I`/`we`, voice labels and placeholder owner
names cannot be used as `owner_name`.

`deadline_text` is retained from the cited transcript. `deadline_date` is
accepted only when that wording contains one explicit, unambiguous date in a
supported format (ISO, English/Italian month names or unambiguous numeric
forms). Relative dates such as `next Friday`, ambiguous numeric dates and
missing deadlines remain without a normalized date. Validation receives no
upload timestamp and never derives a deadline from one.

These are deterministic traceability checks, not semantic verification. A
valid segment ID proves only that the cited segment exists. The test suite
deliberately includes a decision whose wording contradicts its cited segment;
the export contains no “verified” status. Summary, decision and action-item
meaning still requires human review, and names are not verified as real-world
identities.

## Verification

- Focused minutes tests: **28 passed**.
- Default suite: **233 passed, 30 skipped**, 13.83 seconds; one existing
  Starlette/AnyIO deprecation warning.
- Ruff, formatting, mypy, lockfile and whitespace checks passed.
- No real-model, browser, GPU or all-opt-in integration test was run for this
  schema-only slice. The skipped integration cases do not establish minutes
  quality or transcript accuracy.

Known ASR exact-text failures and the repeated-sentence XFAIL were not changed.
See [test status](../status/test-status.md) for their original results.

## Changed files

- `app/minutes_contract.py` — typed response and export-v1 models.
- `app/minutes_validation.py` — parsing, completeness/evidence checks and
  transcript-to-export assembly.
- `test/unit/test_minutes_contract.py` and
  `test/unit/test_minutes_validation.py` —
  contract, malformed/incomplete response, citation, owner, deadline and
  export cases.
- `pyproject.toml` and `uv.lock` — direct Pydantic runtime pin.
- `plan.md`, `README.md` and `docs/status/test-status.md` — current state and results.

**Next:** Sprint 3.4 adds bounded summary extraction and prompt/runtime
controls. The existing read-only transcript UI remains temporary until slice
3.5; this slice neither exposes transcript segments in new UI nor adds
transcript editing.
