# Sprint 3.5 — Edit and export draft minutes

## Outcome

Added a temporary, end-to-end meeting-minutes workflow. With the optional
local Ollama profile enabled, users can generate minutes after transcription,
edit the summary, topics, decisions, action items, owners and deadlines, save
the current revision, mark it reviewed, and download TXT or JSON.

The front page displays and edits summary content only. Normal transcription
status responses no longer include the transcript result. A low-prominence
footer link opens a separate page for an explicit transcript TXT download;
that page never renders transcript text. Transcript export remains read-only.

## Editing, review and export

Minutes generation runs as a cancellable asynchronous task. The coordinator
keeps at most 16 temporary results and uses a one-hour lifetime from minutes
completion. Unlike transcription results, minutes expiry is checked during
coordinator operations, not by a background expiry thread. Minutes are also
removed when their corresponding transcription job is no longer retained.
Transcription results expire one hour after transcription completes, so a
summary generated late may become unavailable before its own one-hour lifetime
ends; API access also requires the transcription job to remain available.
This possible mismatch is inferred from the current lifecycle code and has
not been reproduced in a test. Revisit and add a regression test when changing
retention behavior. Both minutes and transcripts are in memory only and are
lost on refresh or restart. The uploaded audio is removed after transcription;
this slice does not implement saved audio, persistent meetings, manual
deletion or seven-day expiry.

Edits use an expected minutes revision, so stale saves fail instead of
overwriting newer edits. A successful edit increments the revision and clears
the reviewed marker; only the current saved revision can be marked reviewed.
Exports derive their review status and revision fields from this same state.
TXT contains the minutes and status only. **JSON export v1 intentionally
contains the source transcript segments, timestamps and anonymous speaker
IDs** as required by the existing contract; the browser downloads the JSON
attachment without rendering its transcript content.

Generated minutes retain the shape, completeness, citation and evidence checks
from slices 3.3–3.4. Human edits still validate their structure, dates and any
citation IDs supplied, but may add an item without citations and do not receive
the model-evidence checks. Such edits remain drafts until the user explicitly
marks them reviewed; review is a user assertion, not automated fact checking.

## Validation

- Focused minutes, API, browser-asset and upload tests: **54 passed**.
- Opt-in Chromium browser flow: **8 passed** in 11.07 seconds.
- Default suite: **264 passed, 28 skipped** in 14.54 seconds, with one existing
  Starlette/AnyIO deprecation warning.
- Ruff, formatting, mypy, lockfile, whitespace, JavaScript syntax and Compose
  configuration checks passed.
- No real Ollama minutes extraction or full GPU/audio-to-minutes run was
  performed. The template-token reserve and runtime token counts still need a
  real extraction check; slice 3.7 is the planned model-quality checkpoint.
  The opt-in GPU/audio-to-minutes test was not run.

The known two-voice ASR exact-text mismatch (`Friday` → `Fragi`, `Elena` →
`Elina`) and repeated-sentence XFAIL are unchanged. This slice does not
establish transcription or minutes accuracy on real meetings.

## Main implementation files

- `app/core/meeting_minutes.py` — transient task lifecycle, editing, review and
  versioned export assembly.
- `app/main.py` — sanitized job-status responses and minutes/transcript routes.
- `app/core/minutes_validation.py` — generated-output checks and manual-edit
  validation.
- `app/web/index.html`, `app/web/app.js` and `app/web/styles.css` — summary-only
  editor, review and exports.
- `app/web/transcript.html` and `app/web/transcript.js` — separate transcript
  download page.
- `test/unit/test_meeting_minutes.py`, `test/unit/test_main.py`,
  `test/unit/test_browser_assets.py`, `test/unit/test_uploads.py` and
  `test/integration/test_browser_flow.py` — manager, API, privacy, rendering,
  export and browser-flow coverage.

`README.md`, `plan.md`, `docs/status/test-status.md` and
`docs/architecture/storage-and-scaling.md` now describe the current temporary workflow and
its limits. **Next: slice 3.6, summary regeneration and stale-source
handling.**
