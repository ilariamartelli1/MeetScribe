# Sprint 3.6 — Summary regeneration and stale-source handling

**Current limitation:** Stale-source handling is currently inert in the
user-facing app: there is no transcript-correction or transcription-
regeneration action, and new uploads get new job IDs, so users have no
supported way to make existing minutes stale. The behavior is defensive
groundwork; manager tests exercise it by passing a changed transcript directly
to `MeetingMinutesCoordinator` for the same job ID.

## Outcome

Added explicit minutes regeneration and transcript-revision tracking. A new
source transcript for an existing job increments `transcript_revision`,
invalidates review and marks the current minutes stale. The application does
not regenerate automatically; the browser provides a separate
**Regenerate summary** action.

The old minutes retain the exact attributed transcript revision they cite.
Stale TXT exports identify both the cited and current revisions; stale JSON
exports set `stale` and include the source segments associated with the
minutes. This keeps citations from silently referring to different text when
segment IDs are reused in a replacement transcript. Stale minutes remain
visible as read-only summary content and may be downloaded, but cannot be
marked reviewed.

## Regeneration and concurrency

`POST /api/v1/jobs/{job_id}/minutes/regenerate` accepts an expected minutes
revision. A manual edit sets the record's evidence-check state to false, so
regeneration rejects the request unless `confirm_overwrite` is explicitly
true. The browser always asks before regeneration, including when there are
unsaved edits.

Regeneration runs through the existing cancellable inference path. The current
saved draft remains authoritative while generation is running and is replaced
only after the new extraction succeeds and includes every segment from the
captured source revision. Failure or cancellation preserves the prior minutes;
failure is surfaced separately as `regeneration_error`. Saves, review and
exports are blocked while a regeneration is active. Expected revisions reject
stale saves, and source replacement is rejected while a generation is in
progress. A later source update marks the existing draft stale; only a new
explicit regeneration makes it current.

The UI disables edits and review for stale minutes, displays the source and
current revisions, permits explicitly marked stale exports, and requires a
confirmation dialog before regeneration. Transcript text and speaker labels
remain unavailable for browser display or editing.

## Validation

- Focused manager, API and browser-asset tests: **29 passed**.
- Opt-in Chromium browser flow: **10 passed** in 17.95 seconds.
- Default suite: **271 passed, 30 skipped** in 14.34 seconds, with one existing
  Starlette/AnyIO deprecation warning.
- Ruff, formatting, mypy, lockfile, whitespace and JavaScript syntax checks
  passed.
- No real-model extraction or GPU/audio-to-minutes test was run. The real
  Qwen3 prompt accounting and the quality of generated decisions/actions
  remain unverified; slice 3.7 is the planned model checkpoint.

The known two-voice ASR exact-text mismatch (`Friday` → `Fragi`, `Elena` →
`Elina`) and repeated-sentence XFAIL are unchanged.

## Main implementation files

- `app/meeting_minutes.py` — current/source transcript revisions, explicit
  regeneration, preservation on failure/cancellation and stale exports.
- `app/main.py` — version-checked regeneration endpoint.
- `app/web/index.html` and `app/web/app.js` — regeneration confirmation,
  progress/cancellation and stale-summary behavior.
- `test/unit/test_meeting_minutes.py`, `test/unit/test_main.py`,
  `test/unit/test_browser_assets.py` and `test/integration/test_browser_flow.py`
  — source changes, stale citations, no-overwrite, failure, cancellation and
  browser coverage.
- `README.md`, `plan.md` and `docs/test-status.md` — current behavior, contract
  semantics and validation.

**Next:** slice 3.7 compares a real offline audio-to-minutes run against
human-checked synthetic references and reports quality/resource limitations.
