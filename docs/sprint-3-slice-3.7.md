# Sprint 3.7 — Minutes checkpoint (incomplete)

## Outcome

Added an opt-in offline Docker/GPU evaluation harness for three synthetic
recordings: the English and Italian checkpoints and the known two-voice
ASR-error diagnostic. The harness submits audio through the normal API,
collects the transcript and minutes exports, and is designed to report
extraction-token counts, elapsed times and sampled GPU/host/container memory.
The minutes snapshot now exposes extraction metrics for completed real
generations.

The first approved real run did **not** complete the checkpoint. The English
recording transcribed, but the minutes extractor refused the full transcript
before model inference because it exceeded the measured context budget. The
Italian and diagnostic recordings were not reached. No generated draft,
runtime extraction-token counts, quality comparison or resource report was
produced. Slice 3.7 remains incomplete.

## Observed blocker

The English checkpoint is 297.305 seconds long. The conservative total context
estimate—the prompt upper bound plus reserved generation tokens—was
**11,322 tokens**, above the configured **4,096-token** context.
`MinutesContextBudgetError` was returned and no partial draft was created. This
is the intended fail-fast behavior from slice 3.4, but
the measured case shows that successful transcription of an allowed-length
upload does not guarantee that the current minutes workflow can summarize it.

By user decision, this slice retains fail-fast behavior. Transcript chunking
and a larger-context model are not introduced. A future design change is
required before the over-budget English/Italian checkpoint can pass. The
token value is a conservative bound, not a measured runtime token count;
inference did not begin. No human correction time was measured.

See [minutes context scaling and limitations](minutes-context-scaling.md) for
the current extraction constraints and questions to revisit when resuming 3.7.
See [minutes retrieval options](minutes-retrieval-options.md) for exploratory
chunking and RAG tradeoffs; these do not change the current fail-fast decision.

## Validation

- Focused minutes, API, browser-asset and quality-test gating checks: **29
  passed, 1 skipped**.
- Opt-in Chromium browser flow: **10 passed** in 18.25 seconds.
- Default suite: **271 passed, 31 skipped** in 14.47 seconds, with the
  existing Starlette/AnyIO deprecation warning.
- Ruff, formatting, mypy, lockfile, whitespace, JavaScript syntax and Compose
  configuration checks passed.
- Real offline Docker/GPU quality run: **failed** after 322.69 seconds at the
  context-budget check described above. The Compose containers were cleaned up.

The previous exact-text two-voice mismatch (`Friday` → `Fragi`, `Elena` →
`Elina`) and repeated-sentence XFAIL were not rerun by this attempt. The
known-ASR-error case was not reached. No claim about generated-minutes accuracy
or longer-term VM capacity can be made from this run.

## Main implementation files

- `app/meeting_minutes.py` — extraction metrics included in minutes snapshots.
- `app/main.py` — API serialization of those metrics.
- `compose.minutes-quality.yaml` — read-only audio fixture mounts.
- `test/src/minutes_quality_smoke.py` — API-driven transcription, minutes and
  resource-report harness.
- `test/integration/test_minutes_quality.py` — opt-in offline Compose/GPU test.

`README.md`, `plan.md` and `docs/test-status.md` record the command, failure and
known limitation. All work remains uncommitted and unpushed.
