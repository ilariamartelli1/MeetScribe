# Minutes quality evaluation

## Trial summary

- **Prompt-only revisions:** Results varied across cases; action, decision and
  deadline errors remained.
- **Structured extraction plus a second LLM summary call:** The follow-up
  generation could add unsupported claims. A prototype returned owner `A`,
  treated `After the meeting` as a deadline, and added a false Italian decision.
- **Experimental branch approach:** One structured LLM analysis call, followed by
  source-aware deterministic normalization and summary assembly. It keeps
  Marco's in-meeting description out of follow-up actions, retains the
  requested results-review action, and avoids treating project status as a
  decision. One strong, unique weekday match in an explicit deadline phrase
  becomes the confirmed minutes deadline; the transcript remains unchanged.
- **Observed results and limits:** Earlier single runs passed 12/12
  fixed-transcript checks and 6/6 two-voice checks. The 308-unit and filtered
  two-voice results below are from branch `llm-normalize-python-summary`, not
  `main`; that branch remains unreviewed. The latest two-voice run verifies
  `on Friday` in the action deadline and summary while ASR still outputs
  `Fragi`; `Elina` remains uncorrected. The 297-second checkpoint still exceeds
  the 4,096-token context. The duplicate fixed-text and two-voice test runners
  were later removed; the standard app-backed quality test now supports
  selecting the short two-voice case.

This report records human-reviewed, single-run results, not a general model
benchmark. The six-case fixed-transcript evaluator and its inventory were
retired on 2026-11-22 to reduce duplicate test infrastructure; detailed
results below remain historical.

The fixed cases covered in-meeting activities versus follow-up tasks, named
owners, relative deadlines, decisions versus status, English and Italian, and
the risk of mistaking a real name for an ASR error. In particular, the user
says Marco's description was an in-meeting activity in the two-voice case and
the results review was a follow-up action. Neither activity's timing is
explicit in the transcript alone; reviewers must not assume the model could
know it.

## Reported two-voice baseline

The user supplied one transcript and its minutes as text. No raw model response,
citations, runtime metrics or repeated runs were supplied. The observed minutes:

```text
DRAFT / UNREVIEWED

Summary
This is a test meeting where the project is on schedule, with a review of results on Fragi and Marco describing latest changes to the service.

Topics
- Project status review

Decisions
- Project is on schedule

Action items
- Review results on Fragi | Owner: Not specified | Deadline: Not specified
- Describe latest changes to the service | Owner: Not specified | Deadline: Not specified
```

In the user's output, `Fragi` propagates into the summary and action description;
the deadline field remains unspecified. The
[source script](../../test/data/smoke/two-voices-script.txt) says `Friday` and
`Elena`, while the reported transcript says `Fragi` and `Elina`. The status
was listed as a decision. Marco's in-meeting description was listed as an
action without an owner; even if considered an action, the sentence names
Marco. The user clarified that reviewing the results is a post-meeting action;
the transcript alone does not establish that timing.

## Real-app two-voice baseline

An approved offline Docker/GPU run on 2026-11-22 collected the same synthetic
audio through the app and generated minutes with `qwen3:4b`. Collection passed;
four quality checks failed in **80.01 seconds** total:

- The minutes retained `Fragi` rather than `Friday`.
- Marco's description appeared as a follow-up action.
- That action had no named owner.
- The project being on schedule appeared as a decision.

After that run, a Friday-deadline check was added and the saved output was
replayed without model inference: **1 passed, 5 failed** in 0.08 seconds.
The added check failed because the results-review action had no deadline.
After the user clarified that reviewing the results is a required follow-up,
a separate presence check was added. Replaying the same output with that check
gave **2 passed, 5 failed** in 0.09 seconds; the required-action check passed.

This run transcribed in 20.079 seconds and generated minutes in 53.216 seconds.
The estimated prompt upper bound was 2,357 tokens; the runtime reported 526
prompt tokens and 238 generated tokens. These are one run's observations,
not accuracy or capacity estimates. The generated transcript and minutes
matched the user-reported output above. The full machine-readable artifact is
gitignored at `data/evaluations/two-voices-1671144d752d47b1b4c40057106c1ba8.json`.

The standalone two-voice wrapper and its replay support were later removed.
The existing real-app quality integration test now collects transcript and
minutes for all audio cases and can select `known-asr-error` alone; see the
[README](../../README.md). Historical artifacts from earlier runs remain
gitignored local files and are not replayed by the current test.

The checks deliberately target this fixture, not arbitrary paraphrases or
general semantic accuracy. The owner check is conditional: it passes when the
description is absent, and requires Marco when that specific action is
present. The in-meeting expectation uses the user's external clarification;
the transcript alone does not establish timing. The results-review action is
also a user-specified expectation, although the transcript alone does not
establish that it is post-meeting work. Compare fixed-input runs separately to
isolate minutes generation from ASR variation.

The deadline evidence validator checks whether deadline wording occurs in the
**cited** transcript segments, ignoring case and whitespace differences. It
still rejects unsupported `Friday` when the cited segment says `Fragi`; the
extractor now intercepts a sufficiently similar, uniquely closest weekday
candidate, clears it from confirmed deadline fields and records a tentative
note instead. A word
identified as a name or product term is not proposed as a weekday. A textual
match alone does not establish that a word is an accurate deadline.

## Experimental branch approach: one structured call, deterministic post-processing

An earlier prototype made two LLM requests: first, structured analysis;
second, prose summary generation from that analysis. The second request could
add claims not supported by the validated data. The branch implementation uses
one structured analysis request, then deterministic, evidence-aware
normalization and summary assembly. Post-processing demotes unsupported
decision labels to source-cited topics, drops unsupported owners, separates
anonymous-speaker ownership from named owners, removes meeting-relative
phrases such as `After the meeting` from deadline fields, and restores the
exact source span for recognized weekday deadlines. The original transcript
is never edited.

In this earlier revision, possible weekday corrections were restricted to an
unknown word in a time-preposition phrase, compared against weekdays in the
transcript's language, and emitted only when `SequenceMatcher` similarity was
at least `0.5` and the best candidate led the next candidate by at least
`0.15`. The original word remained visible beside an explicitly unverified
candidate; neither candidate nor date was used as a confirmed deadline. This
policy was superseded by the reviewed normalization revision below.

On 2026-11-22, the fixed-transcript Ollama evaluation completed **12 passed**
in **78.68 seconds** across all six cases using `qwen3:4b` at the pinned digest.
Each case used one generation; the artifacts record raw outputs and per-call
token counts in
`data/evaluations/minutes-prompt-f916b07358f44bcc86b09212b8515781.json`
(gitignored). The real audio-to-minutes two-voice run completed **6 passed**
in **79.36 seconds**. Its ASR transcript still contained `Fragi` and `Elina`;
the minutes retained both source forms, showed `Fragi` → `Friday` only as
unverified, left the review deadline null, kept the results review as an
action, excluded Marco's in-meeting description from actions, and kept project
status out of decisions. The audio artifact is
`data/evaluations/two-voices-ecd4beccb5db48aebe7c6f0bb2b276cb.json`
(gitignored). The current unit suite passed **326 tests** with one existing
Starlette/AnyIO deprecation warning; targeted Ruff, formatting and mypy checks
also passed.

These are single-run results on the development machine, not a general quality
benchmark. The fixed-text suite does not measure ASR. The real-audio run
confirms the transcript errors were not fixed, and the known exact-text ASR
failure remains outside this minutes-quality change. The long 297-second
checkpoint still exceeds the configured context and remains a separate,
unresolved Sprint 3.7 limitation.

## Experimental weekday normalization (2026-11-22)

The user rejected a tentative correction note and requested a confirmed
weekday in the minutes and action deadline, while keeping the original
transcript unchanged. This behavior is experimental on
`llm-normalize-python-summary`; it is not in `main` and has not been accepted
as the project flow. The branch extractor applies a shared deterministic
normalizer and validator to explicit English (`on`, `by`, `before`, `until`)
and Italian (`entro`, `per`) deadline phrases. A correction is accepted only
when the best weekday similarity is at least `0.5`, leads the next candidate
by at least `0.15`, and no named-term safeguard matches. The source phrase
becomes (for example) `on Friday` in the deadline and related minutes text;
the summary includes it without a caveat. Ambiguous or weak matches are not
corrected, and a weekday never becomes an inferred calendar date.

The updated unit suite passed **308 tests** in **14.79 seconds** with one
existing Starlette/AnyIO deprecation warning. The real two-voice quality test
passed **1 integration test** in **77.59 seconds**. It verified that the ASR
transcript still contains `Fragi`, while the action deadline and summary use
`on Friday` and the minutes contain no unverified correction note. Unit tests
also cover an independent English typo, an Italian weekday typo, weak matches
and named-term counterexamples. These remain single-run development-machine
results; they do not establish improved ASR accuracy, person-name correction,
or support for the over-budget 297-second meeting.

## Review procedure if a fixed-text comparison is reintroduced

1. Use the same input segments, speaker labels, generation settings and
   response format for every approach being compared. Record the model and
   prompt versions, configuration, run count and any failures. Reuse the
   reviewed transcript rather than rerunning ASR; this isolates minutes
   generation from ASR variability. Record when timestamps are synthetic.
2. Keep the unmodified model outputs, including citations and null fields.
   Record each case's extraction or validation failure separately rather than
   treating it as an empty successful output. Protect any non-synthetic
   recordings and outputs; these cases use synthetic or reported test data.
3. Review each case's checks against both its input and reference notes. For
   yes/no checks record pass, fail or indeterminate with a short rationale and
   supporting segments. The earlier fixed `Fragi` check was automated but
   fixture-specific; other errors such as `Elina` remain descriptive. Do not
   generalize a passing word-specific check to ASR accuracy. An owner check
   conditional on an action being present is not applicable when the action
   is absent.
4. Report per-case outcomes, failures and indeterminate or not-applicable
   checks before any aggregate. Separate unsupported claims, missed facts,
   owner/deadline errors and propagation of ASR errors. A correct-looking
   `Friday` is not proof that a text-only model can reliably correct audio;
   silent confident changes to names or deadlines can also be wrong.

The transcript-only evaluator, its fixture-specific checker tests and the
separate two-voice runner were retired on 2026-11-22. Current checks are in the
standard real-app quality test and the provider-independent extractor unit
tests. The fixed-transcript results above are historical and cannot be replayed
through a dedicated evaluator. No general quality threshold has been selected.
Future model-run or prompt/correction changes still require separate approval.
