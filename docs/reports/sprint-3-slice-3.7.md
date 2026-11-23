# Sprint 3.7 — Minutes checkpoint (incomplete)

## Outcome

Added an opt-in offline Docker/GPU evaluation harness for three synthetic
recordings: the English MP3 and Italian checkpoints and the known two-voice
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

The initial run retained fail-fast behavior; no transcript chunking or model
change was introduced. The token value is a conservative bound, not a measured
runtime token count; inference did not begin. No human correction time was
measured. The later context and output-limit update is recorded below.

See [minutes context scaling and limitations](../investigations/minutes-context-scaling.md) for
the current extraction constraints and questions to revisit when resuming 3.7.
See [minutes retrieval options](../architecture/minutes-retrieval-options.md) for exploratory
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

## English WAV checkpoint follow-up (2026-11-23)

The manifest context was raised to **12,288 tokens** and the output limit to
**1,024 tokens**. A first retry at 12,288 context with the old 768-token output
limit stopped with `reason=length` after exactly 768 generated tokens; this was
not a context-budget rejection. The English-only test then passed **1
integration test** in **126.09 seconds**.

For the 297.305-second fixture, the prompt upper bound was **10,290 tokens**,
while Ollama reported **2,901 prompt tokens** and **812 generated tokens**.
Transcription took **53.767 seconds**, and minutes generation took **67.259
seconds**. The sampled GPU peak was **4,444 MiB** during minutes generation
and **2,294 MiB** during transcription, out of **6,141 MiB** total; no GPU
error occurred. This confirms a complete response for the English fixture but
does not score its semantic quality. The Italian and known-ASR-error cases
were not run with the new settings, and slice 3.7 remains incomplete.

## English MP3 checkpoint follow-up

The English `english-checkpoint` case now submits `meeting-en.mp3`; the
resource harness reads its duration through PyAV and sends it with the MP3
media type. The manifest context was raised to **16,384 tokens** and the
generation limit to **4,096 tokens**. The selected integration case passed in
**203.09 seconds** for the 297.305-second recording.

The prompt upper bound was **10,199 tokens**; Ollama reported **2,862 prompt
tokens** and **1,023 generated tokens**. Transcription took **72.659 seconds**
and minutes generation **120.283 seconds**. Sampled GPU use peaked at **4,710
MiB** during minutes generation and **2,294 MiB** during transcription, out of
**6,141 MiB** total; no GPU error occurred. This confirms a complete response
for this MP3 fixture, not semantic quality or a general context-capacity
limit. The Italian and known-ASR-error cases remain untested with the current
settings, so slice 3.7 remains incomplete.

## Main implementation files

- `app/core/meeting_minutes.py` — extraction metrics included in minutes snapshots.
- `app/main.py` — API serialization of those metrics.
- `compose.minutes-quality.yaml` — read-only audio fixture mounts.
- `test/src/minutes_quality_smoke.py` — API-driven transcription, minutes and
  resource-report harness.
- `test/integration/test_minutes_quality.py` — opt-in offline Compose/GPU test.

`README.md`, `plan.md` and `docs/status/test-status.md` record the initial
failure, follow-up run and remaining limitations. All work remains uncommitted
and unpushed.
