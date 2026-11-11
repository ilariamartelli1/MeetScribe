# Sprint 1 checkpoint: previous-text conditioning enabled

Date: 2026-11-11. This is a separate experiment, not first-release acceptance.
The [previous checkpoint report](sprint-1-checkpoint.md) describes the run with
conditioning disabled.

## Setup

For this experiment only, `CONDITION_ON_PREVIOUS_TEXT` was changed from `False`
to `True`. The production adapter and its provenance both used the enabled
setting. The model revision, GPU (`RTX 4050`, `int8_float16`), integrated VAD,
beam size 5 and synthetic checkpoint files were unchanged. Inference used the
provisioned model inside a Docker container with networking disabled.

The test suite was run without changing its assertions. A separate offline
adapter run then captured raw segments for English and Italian WAV and MP3;
each captured transcript was passed through the production service's validation
without running inference a second time. These are separate runs, so their
individual outputs need not match exactly. The reference scripts and generated
timings remain incompletely human-verified.

## Test suite

With conditioning enabled, each of three full suite runs including opt-in
browser, Docker and GPU tests reported **113 passed, 3 failed**:

- The long English MP3 regression failed because `TranscriptionService`
  rejected the result with `InvalidTranscriptError`. The error reports invalid
  segment text or ordering/bounds, but that test did not capture the rejected
  raw segments, so the precise invalid segment in each of those runs is unknown.
- Two adapter unit tests failed because their assertions explicitly require
  `condition_on_previous_text=False`. Those failures are expected for this
  intentional configuration change and are not, by themselves, evidence of
  incorrect transcription. The out-of-bounds unit test did verify rejection
  before reaching its setting assertion.

The browser tests use fake inference. The feasibility and short-audio accuracy
tests call faster-whisper directly, rather than exercising the production
adapter's conditioning setting.

## First raw-output comparison

The table compares the captured enabled-conditioning run with the previous
disabled-conditioning adapter run on the same recordings. Durations are
original-audio durations; maximum end times are model output. Each row represents
one run, not a statistical benchmark.

| Recording | Duration | Enabled: service result / segments / maximum end | Earlier disabled: service result / segments / maximum end |
|---|---:|---|---|
| English WAV | 297.305 s | Accepted / 49 / 296.800 s | Accepted / 50 / 296.940 s |
| English MP3 | 297.305 s | **Rejected** / 52 / 316.760 s | Accepted / 50 / 296.940 s |
| Italian WAV | 73.318 s | Accepted / 12 / 72.900 s | Accepted / 12 / 72.900 s |
| Italian MP3 | 73.318 s | Accepted / 12 / 72.900 s | Accepted / 12 / 72.900 s |

In the enabled MP3 capture, the expected final sentence was present, followed
by three additional segments that extended beyond the recording:

| Model timestamps | Model text |
|---|---|
| 296.800–300.660 s | `team members in the meeting notes, are tuned.` |
| 300.820–315.680 s | `mental pain` |
| 315.800–316.760 s | `Music` |

The last segment ended **19.455 seconds** after the recording ended. The service
rejected the complete transcript; it did not clamp timestamps or return a
success-shaped partial result. These segments came from the separate capture,
not the regression test's uncaptured failure.

Each of the 11 English and four Italian generated quiet-word intervals overlapped
an enabled-run segment containing that word. This does not establish word-level
timing or independently verify repeated words within a segment.

## Differences from the draft scripts

The following comparisons ignore case and punctuation but do not normalize
numbers. Currency symbols were checked in the full text. Times are approximate
generated sentence intervals, not human-verified word timestamps.

| Audio | Draft interval | Type | Reference and observed output | Captured runs |
|---|---|---|---|---|
| EN WAV and MP3 | 00:13.681–00:18.950 | Name | `Elena` → `Elina` | 3/3 each |
| EN WAV and MP3 | 00:25.241–00:31.813 | Format spelling | `wave` → `WAV` | 3/3 each |
| EN WAV | 00:32.163–00:38.077 | Spelling/spacing | `timestamps` → `time stamps` | 3/3 |
| EN WAV and MP3 | 00:32.163–00:38.077 | Article substitution | `a reviewer` → `the reviewer` | 3/3 each |
| EN WAV and MP3 | 01:02.560–01:07.912 | Date formatting | `fourteen April` → `14 April` | 3/3 each |
| EN WAV and MP3 | 01:14.902–01:20.340 | Amount formatting | `two thousand four hundred euros` → `€2,400` | 3/3 each |
| EN WAV and MP3 | 03:51.266–03:57.008 | Date formatting | `twenty seven May` → `27 May` | 3/3 each |
| EN WAV and MP3 | 03:57.358–04:04.115 | Number formatting | `fifteen` → `15` | 3/3 each |
| EN MP3 | After the final sentence | Extra content | Different extra content in each capture; see below | 3/3 |
| IT WAV and MP3 | 00:29.685–00:36.541 | Date and amount formatting | `quattordici` → `14`; `duemila quattrocento` → `2400` | 3/3 each |
| IT WAV and MP3 | 01:07.704–01:12.968 | Date formatting | `ventisette` → `27` | 3/3 each |

The English voice's spoken `colon` is included in the draft reference and in
the enabled transcripts, so it is not a discrepancy. Number/date formatting and
the currency symbol preserve the apparent values; these comparisons are not WER.
In the previous disabled captures, the English MP3 also rendered a second `Elena` as `Elina`,
and the Italian WAV rendered `riletta` as `diletta`; neither difference appeared
in any of the three enabled captures. Conversely, `a reviewer` → `the reviewer`
appeared in all three enabled English WAV captures but none of the disabled WAV
captures. These differences do not establish a general quality improvement or cause.

## Three-run repeatability

The first offline adapter capture was followed by two more captures for each
recording with the same model, settings, GPU and fixtures. `3/3 each` in the
table means the output appeared in all three captures for each format named.
English WAV, Italian WAV and Italian MP3 had identical segment texts and
timestamps across all three enabled captures, and all passed service validation.
Every generated quiet-word interval overlapped a segment containing that word
in all captures; this is not word-level timing verification.

The English MP3 varied at the end:

| Enabled capture | Service result | Maximum segment end | Extra output after the final sentence |
|---|---|---:|---|
| 1 | Rejected | 316.760 s | Three segments, shown above, ending beyond the 297.305 s recording |
| 2 | **Accepted** | 297.300 s | A 220-word segment at 296.800–297.300 s containing `results of the` 72 times |
| 3 | Rejected | 326.780 s | The final sentence repeated in a segment ending 29.475 s past the recording |

Capture 2 demonstrates that timestamp validation alone does not detect all
incorrect or repetitive text: the entire 220-word segment was within bounds.
For comparison, all three disabled MP3 captures were accepted, had a maximum
end of 296.940 s and contained none of these additional endings. The three
enabled full-suite regression failures were separate inference runs from these
captures; each raised `InvalidTranscriptError`, without preserving raw segments.

These are six captures per recording across two settings, not a representative
benchmark. Formatting differences are not automatically recognition errors,
and consistent output in three captures does not prove a general benefit or
causation.

## Restoration and interpretation

After the experiment, the source setting and built app image were restored to
`condition_on_previous_text=False`. The full suite was rerun with the original
assertions and all opt-in tests: **116 passed**.

The enabled setting coincided with long-MP3 validation failures and incorrect
extra output across the repeated tests and captures. That supports retaining
the current mitigation, but a small number of synthetic runs does not prove the
internal cause or guarantee that disabled conditioning will prevent errors.
Human timing review, representative audio and the other open checkpoint work
remain necessary.
