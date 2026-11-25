# Test status

Snapshot: 2026-11-24. Baseline repository commit: `d92579f`; results from the
current worktree are identified in their rows. This document records observed
results, not a live dashboard. Results apply to the stated environment and
test inputs; reruns may differ.

## CLion test configurations

Inventory checked 2026-11-22. Names are CLion labels; `test/setup` maps each
Python test configuration to its current in-code pytest node name(s) and source
file. The two app/debug launchers are omitted because they do not run tests.
Results and runtimes are historical observations, not fresh runs of every
configuration. For multi-case configurations, runtimes are reported test or
shared-fixture durations, not necessarily total CLion wall time. `Not reported`
means no matching result or duration is documented.

| Name | Test/setup | Reported result | Reported runtime |
|---|---|---|---:|
| Integration - GPU Handoff | [`test/integration/test_gpu_handoff.py`](../../test/integration/test_gpu_handoff.py)::`test_real_gpu_handoff_serializes_and_releases_all_models` | **PASS** ([Slice 3.2](../reports/sprint-3-slice-3.2.md)) | 86.73 s ([Slice 3.2](../reports/sprint-3-slice-3.2.md)) |
| Integration - Minutes Quality | [`test/integration/test_minutes_quality.py`](../../test/integration/test_minutes_quality.py)::`test_real_audio_to_minutes_quality_report` (supports `MEETSCRIBE_MINUTES_QUALITY_CASES`) | Historical English MP3-only **PASS** (16,384 context / 4,096 output; sampled GPU peak 4,710/6,141 MiB); earlier English WAV pass at 12,288/1,024; Italian and known-ASR-error cases not rerun | 203.09 s (MP3); 126.09 s (WAV); 322.69 s (full, historical) |
| Manual - AMI Long-Audio Capacity | Local AMI benchmark; full metrics in [minutes context scaling](../investigations/minutes-context-scaling.md) | 10m44s ES2010a **transcription + minutes PASS** at 26,112; 57m32s EN2001b **transcription PASS**, minutes preflight rejected at 16,384. Not a semantic-quality test or three-hour capacity proof. | 237.38 s (ES2010a at 26,112); 92.91 s and 419.24 s (earlier runs at 16,384) |
| Integration - Ollama Model | [`test/integration/test_ollama_model.py`](../../test/integration/test_ollama_model.py)::`test_pinned_ollama_model_runs_offline_on_gpu` | **PASS** at 26,112 context; Ollama placed 75% on GPU / 25% CPU (26/37 layers on GPU). Historical all-GPU result: [Slice 3.1](../reports/sprint-3-slice-3.1.md) | 59.65 s (26,112); 54.09 s ([Slice 3.1](../reports/sprint-3-slice-3.1.md)) |
| Integration - Browser - Fake Inference | [`test/integration/test_browser_flow.py`](../../test/integration/test_browser_flow.py):<br>`test_real_api_upload_summary_edit_review_and_direct_transcript_download`;<br>`test_regeneration_requires_confirmation_before_replacing_saved_edits`;<br>`test_stale_minutes_are_read_only_until_regenerated`;<br>`test_capacity_confirmation_precedes_replacement` (2 cases);<br>`test_cancel_running_transcription`;<br>`test_cancel_running_minutes_generation`;<br>`test_real_api_cancels_running_fake_transcription`;<br>`test_cancel_error_remains_visible_and_cancel_can_be_retried`;<br>`test_job_failure_is_visible_as_text` | Updated direct-download case **passed**; full configuration last ran in Slice 3.7 (10 passed, using the earlier separate-page case) | Updated case: 4.88 s; Slice 3.6: 17.95 s ([report](../reports/sprint-3-slice-3.6.md)); Slice 3.7 runtime not reported |
| Integration - CPU - ASR and Diarization Feasibility (long) | [`test/integration/test_diarization_feasibility.py`](../../test/integration/test_diarization_feasibility.py)::`test_offline_asr_then_diarization[cpu-None]`; `test_offline_asr_then_diarization[cpu-mp3]` | **PASS**, both CPU cases (2026-11-21 all-opt-in run) | 60.07 s + 24.12 s per case; config wall time not reported |
| Integration - CPU - Text Normalization (long) | [`test/integration/test_wetextprocessing.py`](../../test/integration/test_wetextprocessing.py)::`test_english_and_italian_rule_data` | **PASS** (2026-11-21 all-opt-in run) | 39.79 s |
| Integration - Docker - CPU App | [`test/integration/test_docker.py`](../../test/integration/test_docker.py)::`test_cpu_container_is_healthy_and_bound_to_loopback` | **PASS** (2026-11-21 all-opt-in run) | 3.60 s |
| Integration - Docker - Diarization Device and Arguments | [`test/integration/test_diarization_feasibility.py`](../../test/integration/test_diarization_feasibility.py)::`test_cuda_diarization_fails_without_gpu`; `test_diarization_rejects_nonpositive_run_count` | **PASS**, both checks (2026-11-21 all-opt-in run) | 10.23 s + 10.19 s per case; config wall time not reported |
| Integration - GPU - ASR Checkpoint WER (long) | [`test/integration/test_checkpoint_wer.py`](../../test/integration/test_checkpoint_wer.py)::`test_checkpoint_wer_and_negations` | **PASS** (2026-11-21 all-opt-in run) | 60.07 s |
| Integration - GPU - ASR Feasibility | [`test/integration/test_asr_gpu_feasibility.py`](../../test/integration/test_asr_gpu_feasibility.py)::`test_offline_gpu_inference` | **PASS** (2026-11-21 all-opt-in run) | 8.04 s |
| Integration - GPU - ASR Quiet Negations | [`test/integration/test_asr_adapter.py`](../../test/integration/test_asr_adapter.py)::`test_quiet_negations_and_silence_on_real_model` | **PASS** (2026-11-21 all-opt-in run) | 5.18 s |
| Integration - GPU - ASR Synthetic Accuracy | [`test/integration/test_asr_accuracy.py`](../../test/integration/test_asr_accuracy.py)::`test_synthetic_speech_transcription` | **PASS** (2026-11-21 all-opt-in run) | 5.16 s |
| Integration - GPU - ASR Worker | [`test/integration/test_transcription_jobs.py`](../../test/integration/test_transcription_jobs.py)::`test_real_model_runs_in_an_owned_child` | **PASS** (2026-11-21 all-opt-in run) | 5.06 s |
| Integration - GPU - ASR and Diarization Feasibility | [`test/integration/test_diarization_feasibility.py`](../../test/integration/test_diarization_feasibility.py)::`test_offline_asr_then_diarization[cuda-None]` | **PASS** (2026-11-21 all-opt-in run) | 30.28 s |
| Integration - GPU - Community-1 Adapter | [`test/integration/test_diarization_adapter.py`](../../test/integration/test_diarization_adapter.py)::`test_real_community1_adapter_identifies_two_synthetic_voices` | **PASS** (2026-11-21 all-opt-in run) | 14.32 s |
| Integration - GPU - Speaker Alignment | [`test/integration/test_speaker_alignment.py`](../../test/integration/test_speaker_alignment.py)::`test_offline_job_attributes_alternating_synthetic_voices`;<br>`test_offline_job_produces_segments_and_turns_in_english_and_italian`;<br>`test_offline_job_transcribes_all_two_voice_speech` | Attribution and output-shape checks **PASS**; exact-text check **FAIL** (`Friday` → `Fragi`, `Elena` → `Elina`) | 31.90 s shared English/Italian fixture setup; not full config wall time |
| Integration - Synthetic Two-Voice Fixture | Configured target [`test/integration/test_diarization_adapter.py`](../../test/integration/test_diarization_adapter.py) uses `-k versioned`, which matches no test there. The related in-code fixture check is [`test/unit/test_asr_helpers.py`](../../test/unit/test_asr_helpers.py)::`test_integration_paths_use_versioned_fixtures`, but this config does not target it. | No tests selected by the current target/filter; no result reported | Not reported |
| Integration - GPU - ASR Five-minute MP3 (long) | [`test/integration/test_asr_long_recording.py`](../../test/integration/test_asr_long_recording.py)::`test_five_minute_mp3_has_bounded_timestamps_and_final_speech` | **PASS** (2026-11-21 all-opt-in run) | 10.95 s |
| Unit Tests | All pytest tests under [`test/unit/`](../../test/unit/) (directory target; no single test node) | **285 passed** (provider-independent pipeline update below) | 14.48 s, unit-only command |

**Initial two-voice quality update (before deterministic normalization):** The
focused case, checker and fake-API unit
tests passed **13 tests** in 0.19 seconds. The approved offline Docker/GPU
two-voice run collected the transcript and minutes, then reported **1 passed,
4 failed** in 80.01 seconds. After adding a deadline check, replay of the saved
output reported **1 passed, 5 failed** in 0.08 seconds without model inference.
After adding a required results-review action check, replay reported **2 passed,
5 failed** in 0.09 seconds; the new check passed.
The failures cover `Friday`/`Fragi`, an in-meeting activity classified as a
follow-up, its missing owner, the missing Friday deadline, and project status
classified as a decision. The six integration cases were also confirmed skipped
by default without either opt-in setting. No other integration tests were run. These narrow
checks and their limitations are documented in the
[evaluation notes](../investigations/minutes-quality-evaluation.md).

**Required results-review action update:** Focused case-inventory and
quality-check unit tests passed **13 tests** in 0.08 seconds. The new check
fails if the results review is missing from action items even when it is
mentioned in the summary. No new model run was performed.

**Initial raw deadline evidence update (before safe normalization):** The
minutes-validation and extractor unit tests
passed **45 tests** in 0.14 seconds. They include the two-voice `Fragi` /
`Friday` mismatch, a `Friday` mention in an uncited segment, and extractor
rejection of the unsupported deadline. No model or integration test was run
for this update.

**Direct transcript download update:** The focused browser-assets unit tests
passed **8 tests** in 0.55 seconds with the existing Starlette/AnyIO warning.
The updated fake-inference browser case passed in 4.88 seconds, covering
download errors, successful download and preservation of unsaved summary edits.
The remaining browser cases and real-model, GPU and Docker tests were not run.

**Provider-independent pipeline update:** The unit-only command
`uv run --locked python -m pytest -q test/unit` passed **285 tests** in
14.48 seconds, with one existing Starlette/AnyIO deprecation warning and no
skips. It includes fake-provider and spawned-worker checks, not real-model
execution. Real-model, GPU handoff, browser, Docker and other integration
tests were not run for this refactor; the integration results above remain
historical.

**Application folder grouping update:** The default suite passed **277 tests**
and skipped **31 opt-in cases** in 14.61 seconds, with the existing
Starlette/AnyIO deprecation warning. The added checks cover adapter imports,
shared manifest paths and the container startup module entry point. Ruff,
formatting and mypy passed. No opt-in real-model, Docker, browser or other
integration runs were performed; the historical configuration results above
were not rerun.

**Slice 2.3a update:** After this snapshot, browser editing was removed and the
targeted browser-flow and browser-asset checks passed: 12 tests total, including
8 browser cases and 4 asset checks. The browser test previously named
`test_real_api_upload_edit_and_download` was replaced by
`test_real_api_upload_read_only_and_download`, and a multiple-segment download
case was added. The historical snapshot counts below have not been rerun as a
full suite with these changes; the other integration results remain earlier
observations.

**Slice 2.4 update:** The optional speaker-attributed transcript download was
added. The targeted fake-inference browser flow and browser-asset checks passed
14 tests total (10 browser cases and 4 asset checks), including unknown-speaker,
empty-result, safe-rendering and cancellation behavior. The English/Italian
real-model output-shape check passed; its shared fixture ran both jobs in 31.90
seconds. The separate exact-text check still failed with `Friday` → `Fragi` and
`Elena` → `Elina`, and the repeated-sentence check remained an XFAIL.

**Sprint 3.1 update:** Added the opt-in Ollama/Qwen3 runtime and an offline
real-model smoke. The latest default suite passed 191 tests and skipped 29
opt-in cases in 13.32 seconds; the Docker CPU smoke passed in 3.35 seconds, and
the pinned offline Qwen3 smoke passed in 54.09 seconds. Ruff, formatting, mypy
and all Compose configuration checks passed. The full historical all-opt-in
suite below was not rerun; the model smoke's GPU placement, memory snapshots,
and bounded output settings are recorded in the
[slice 3.1 report](../reports/sprint-3-slice-3.1.md).

**Sprint 3.2 update:** Added the shared GPU inference gate and release-verified
Ollama client. The latest default suite passed **205 tests and skipped 30
opt-in cases** in 14.14 seconds, with one existing Starlette/AnyIO deprecation
warning. The corrected offline real-container handoff test passed in 86.73
seconds; ASR, GPU diarization, Ollama generation/concurrent requests,
cancellation and the subsequent ASR stage all completed. It sampled memory
every 0.5 seconds, observed at most about 0.53 seconds between samples during
active stages, and confirmed 100% GPU placement for Qwen3. The run peaked at
11.1 GiB host RAM and 5.12 GiB of 8.0 GiB swap; swap was already 4.28 GiB at
baseline. See the [slice 3.2 report](../reports/sprint-3-slice-3.2.md) for per-stage
measurements, the preliminary sampler limitation and test caveats. The other
opt-in integration checks and the all-opt-in suite were not rerun.

**Sprint 3.3 update:** Added strict minutes response/export schemas and separate
completeness, citation and evidence checks. The focused contract and validation
tests passed **28 tests**. The latest default suite passed **233 tests and
skipped 30 opt-in cases** in 13.83 seconds, with the same existing
Starlette/AnyIO deprecation warning. The opt-in GPU, Ollama, browser and
accuracy integration tests were not run for this schema-only slice; see the
[slice 3.3 report](../reports/sprint-3-slice-3.3.md) for semantic-validation limits.

**Sprint 3.4 update:** Added a bounded Ollama minutes extractor. Its focused
fake-client tests passed **18 tests**. The default suite passed **251 tests and
skipped 30 opt-in cases** in 13.92 seconds, with the same existing
Starlette/AnyIO deprecation warning; Ruff, formatting, mypy, lockfile and
whitespace checks passed. No real-model extraction or GPU/audio-to-minutes
integration test was run, so the template reserve and measured runtime token
counts remain unverified against an actual extraction. See the
[slice 3.4 report](../reports/sprint-3-slice-3.4.md) for budget details and limitations.

**Sprint 3.5 update:** Added the temporary minutes workflow/API and
summary-only edit/review/export UI. The focused minutes, API, asset and upload
checks passed **54 tests**; the opt-in Chromium browser flow passed **8 tests**
in 11.07 seconds. The default suite passed **264 tests and skipped 28 opt-in
cases** in 14.54 seconds, with the same Starlette/AnyIO deprecation warning.
Ruff, formatting, mypy, lockfile, whitespace, JavaScript syntax and Compose
configuration checks passed. TXT minutes contain only the summary; JSON export
v1 intentionally includes source transcript segments. Manual edits permit
uncited human-added items and are not given the model-evidence validation used
for generated minutes; they remain drafts until reviewed. No real Ollama
minutes extraction or GPU/audio-to-minutes run was performed. The historical
all-opt-in run below predates this slice. See the
[slice 3.5 report](../reports/sprint-3-slice-3.5.md) for privacy, edit and validation
boundaries.

**Sprint 3.6 update:** Added explicit minutes regeneration and source-revision
handling. Changed transcripts mark existing minutes stale and invalidate
review; stale exports retain the exact source segments cited. Regeneration
requires explicit confirmation before overwriting manual edits and keeps the
previous draft if cancelled or failed. The focused manager/API/asset checks
passed **29 tests**; the opt-in Chromium browser suite passed **10 tests** in
17.95 seconds. The default suite passed **271 tests and skipped 30 opt-in
cases** in 14.34 seconds, with the same deprecation warning. Ruff, formatting,
mypy, lockfile, whitespace and JavaScript syntax checks passed. No real-model
extraction or GPU/audio-to-minutes test was run; see the
[slice 3.6 report](../reports/sprint-3-slice-3.6.md). The historical all-opt-in run below
predates slices 3.5–3.6.

**Sprint 3.7 update (incomplete):** Added the opt-in offline audio-to-minutes
evaluation harness and extraction-token metrics. The focused minutes/API/assets
checks passed **29 tests and skipped 1**; the Chromium browser suite passed
**10 tests**. The default suite passed **271 tests and skipped 31** in 14.47
seconds. Ruff, formatting, mypy, lockfile, whitespace, JavaScript syntax and
Compose configuration checks passed.

The approved real Docker/GPU run failed after **322.69 seconds**. The
297.305-second English checkpoint transcribed successfully, but minutes
generation returned `MinutesContextBudgetError`: the conservative total
context estimate (prompt upper bound plus reserved generation tokens) was
**11,322 tokens versus a 4,096-token context**. The failure occurred before
model generation and correctly produced no partial draft. The Italian and
known-ASR-error recordings were not reached; there are no real extraction-token
counts or resource report from that run. Human correction time was not
measured. Fail-fast behavior remains in place for transcripts that exceed the
configured context; no transcript chunking or model change was introduced. The
configured upload limit does not guarantee that every transcript can be
summarized; see the
[slice 3.7 report](../reports/sprint-3-slice-3.7.md).

**Minutes quality iteration (2026-11-22):** Replaced the second LLM summary
call with a single structured analysis call followed by deterministic,
source-aware normalization and summary assembly. The fixed-transcript Ollama
evaluation passed **12 tests** across six cases in **78.68 seconds**. The real
Docker/GPU two-voice audio-to-minutes run passed **6 tests** in **79.36
seconds**. Its ASR transcript still contained `Fragi` and `Elina`; the minutes
preserved the transcript, presented `Fragi` → `Friday` only as unverified, left
the results-review deadline null, retained that requested action, kept Marco's
description out of action items, and did not label project status as a
decision. The run used **626 prompt tokens** and **259 generated tokens** for
minutes after a **19.581-second** transcription; the extraction prompt bound
was **2,827** tokens. It is not evidence that ASR accuracy improved or that a
name correction is safe. The artifacts are gitignored at
`data/evaluations/minutes-prompt-f916b07358f44bcc86b09212b8515781.json` and
`data/evaluations/two-voices-ecd4beccb5db48aebe7c6f0bb2b276cb.json`.

**Quality test consolidation (2026-11-22):** Removed the separate frozen-text
evaluator and two-voice integration wrapper. The existing real-app quality
test now runs all three audio cases by default or selects a case with
`MEETSCRIBE_MINUTES_QUALITY_CASES`; it still collects and prints the actual
transcript and minutes. Its filtered `known-asr-error` run passed **1
integration test** in **77.16 seconds**. The output again contained `Fragi`
and `Elina` in the ASR transcript; the minutes kept `Fragi` → `Friday`
unverified, left the review deadline null, retained the review action, omitted
Marco's in-meeting description as an action, and excluded project status from
decisions. The earlier 12-check and 6-check results above are historical runs
from before consolidation.

After removing test-harness-only unit modules, the provider-independent suite
passed **301 tests** in **14.43 seconds** with one existing Starlette/AnyIO
deprecation warning. Targeted Ruff, formatting and mypy checks passed. The full
integration suite, long
297-second minutes-context test, and separate exact-text ASR test were not run
in this iteration. At that point, slice 3.7 remained incomplete; the later
English-only context rerun is recorded below.

**All-opt-in test run:** On 2026-11-21, the complete suite ran with all opt-in
flags enabled using:

```sh
MEETSCRIBE_ASR_TESTS=1 MEETSCRIBE_BROWSER_TESTS=1 MEETSCRIBE_DOCKER_TESTS=1 \
MEETSCRIBE_TEXT_NORMALIZATION_TESTS=1 \
uv run --locked python -m pytest -q -r a --durations=0
```

Result: **217 passed, 1 failed, 1 XFAIL, 0 skipped** in 352.51 seconds
(5:52.51), with one Starlette/AnyIO deprecation warning. The failure was the
known two-voice exact-text mismatch above. The real-model English/Italian
speaker-attributed fixture setup took 31.90 seconds for both jobs; pytest
attributes shared module-fixture setup to its first test.

| Slow test or setup | Observed duration |
|---|---:|
| `test_checkpoint_wer.py::test_checkpoint_wer_and_negations` | 60.07 s |
| `test_diarization_feasibility.py::test_offline_asr_then_diarization[cpu-None]` | 60.07 s |
| `test_wetextprocessing.py::test_english_and_italian_rule_data` | 39.79 s |
| `test_speaker_alignment.py` shared English/Italian fixture setup | 31.90 s |
| `test_diarization_feasibility.py::test_offline_asr_then_diarization[cuda-None]` | 30.28 s |
| `test_diarization_feasibility.py::test_offline_asr_then_diarization[cpu-mp3]` | 24.12 s |
| `test_diarization_adapter.py::test_real_community1_adapter_identifies_two_synthetic_voices` | 14.32 s |
| `test_asr_long_recording.py::test_five_minute_mp3_has_bounded_timestamps_and_final_speech` | 10.95 s |
| `test_diarization_feasibility.py::test_cuda_diarization_fails_without_gpu` | 10.23 s |
| `test_diarization_feasibility.py::test_diarization_rejects_nonpositive_run_count` | 10.19 s |
| `test_asr_gpu_feasibility.py::test_offline_gpu_inference` | 8.04 s |
| `test_asr_adapter.py::test_quiet_negations_and_silence_on_real_model` | 5.18 s |
| `test_asr_accuracy.py::test_synthetic_speech_transcription` | 5.16 s |
| `test_transcription_jobs.py::test_real_model_runs_in_an_owned_child` | 5.06 s |
| `test_asr_repetition.py::test_repeated_sentence_on_real_model` | 4.74 s |
| `test_docker.py::test_cpu_container_is_healthy_and_bound_to_loopback` | 3.60 s |

These are single-run observations on the development host, not stable
benchmarks. `--durations=0` hides test phases below 5 ms; other unit and browser
cases were substantially shorter than the slowest integration checks. The
historical 24-integration-case totals below do not include this new opt-in case.

## Summary

- **Default suite:** 191 passed, 24 skipped, 0 failed. All 191 executed tests
  passed; the 24 opt-in integration cases were not executed in this run.
- **Selected real-model checks:** the checkpoint, quiet-negation, five-minute
  timestamp and speaker-attribution checks passed. The two-voice transcript
  accuracy check failed; the repeated-sentence check was an expected failure
  (XFAIL).
- **Remaining integration checks:** all 18 previously unvalidated cases
  passed in a separate opt-in run. Across the targeted runs, the 24 integration
  cases have 22 passes, 1 failure and 1 XFAIL. These results came from multiple
  runs, not one full-suite invocation.

The default suite emitted one Starlette/AnyIO deprecation warning. The
production-adapter GPU checks used the explicitly configured 30-second ASR
window. Some feasibility tests use the model directly, and diarization
feasibility also tests CPU. Synthetic fixtures do not establish accuracy on
real meetings.

## Default suite

Command, with all integration opt-in environment variables unset:

```sh
python -m pytest -q -r s
```

The 191 passed unit cases are listed by file below. Counts include individual
parameterized cases.

| Unit test file | Passed / executed |
|---|---:|
| `test_asr_contract.py` | 2 / 2 |
| `test_asr_evaluation.py` | 12 / 12 |
| `test_asr_feasibility.py` | 8 / 8 |
| `test_asr_helpers.py` | 3 / 3 |
| `test_asr_image.py` | 2 / 2 |
| `test_asr_long_recording_regression.py` | 3 / 3 |
| `test_asr_repetition.py` | 6 / 6 |
| `test_asr_service.py` | 11 / 11 |
| `test_audio_fixtures.py` | 5 / 5 |
| `test_browser_assets.py` | 4 / 4 |
| `test_community1_diarizer.py` | 14 / 14 |
| `test_config.py` | 1 / 1 |
| `test_container_start.py` | 10 / 10 |
| `test_diarization_contract.py` | 1 / 1 |
| `test_diarization_service.py` | 9 / 9 |
| `test_faster_whisper_asr.py` | 14 / 14 |
| `test_main.py` | 5 / 5 |
| `test_model_integrity.py` | 11 / 11 |
| `test_speaker_alignment.py` | 4 / 4 |
| `test_speaker_alignment_regression.py` | 6 / 6 |
| `test_transcription_jobs.py` | 23 / 23 |
| `test_transcription_worker.py` | 5 / 5 |
| `test_uploads.py` | 32 / 32 |

All paths in this table are under `test/unit/`.

## Integration tests

All 24 cases below were skipped in the default run. The separate observed
results are from targeted opt-in runs on 2026-11-21. Paths are under
`test/integration/`.

| Test | Cases | Latest observed status |
|---|---:|---|
| `test_checkpoint_wer.py::test_checkpoint_wer_and_negations` | 1 | **PASS**; all four recordings met the WER, deletion and negation gates |
| `test_asr_adapter.py::test_quiet_negations_and_silence_on_real_model` | 1 | **PASS** |
| `test_asr_long_recording.py::test_five_minute_mp3_has_bounded_timestamps_and_final_speech` | 1 | **PASS** |
| `test_speaker_alignment.py::test_offline_job_attributes_alternating_synthetic_voices` | 1 | **PASS** |
| `test_speaker_alignment.py::test_offline_job_transcribes_all_two_voice_speech` | 1 | **FAIL**; exact normalized text differs on `Friday` and `Elena` |
| `test_asr_repetition.py::test_repeated_sentence_on_real_model` | 1 | **XFAIL**; one copy of the repeated sentence was omitted, as expected by the diagnostic |
| `test_asr_accuracy.py::test_synthetic_speech_transcription` | 1 | **PASS** |
| `test_asr_gpu_feasibility.py::test_offline_gpu_inference` | 1 | **PASS** |
| `test_diarization_adapter.py::test_real_community1_adapter_identifies_two_synthetic_voices` | 1 | **PASS** |
| `test_diarization_feasibility.py::test_offline_asr_then_diarization` | 3 | **PASS**: CUDA default, CPU default, CPU MP3 |
| `test_diarization_feasibility.py::test_cuda_diarization_fails_without_gpu` | 1 | **PASS** |
| `test_diarization_feasibility.py::test_diarization_rejects_nonpositive_run_count` | 1 | **PASS** |
| `test_transcription_jobs.py::test_real_model_runs_in_an_owned_child` | 1 | **PASS** |
| `test_docker.py::test_cpu_container_is_healthy_and_bound_to_loopback` | 1 | **PASS** |
| `test_browser_flow.py::test_real_api_upload_edit_and_download` | 1 | **PASS** |
| `test_browser_flow.py::test_capacity_confirmation_precedes_replacement` | 2 | **PASS**: confirm and decline |
| `test_browser_flow.py::test_cancel_running_job` | 1 | **PASS** |
| `test_browser_flow.py::test_real_api_cancels_running_fake_job` | 1 | **PASS** |
| `test_browser_flow.py::test_cancel_error_remains_visible_and_cancel_can_be_retried` | 1 | **PASS** |
| `test_browser_flow.py::test_job_failure_is_visible_as_text` | 1 | **PASS** |
| `test_wetextprocessing.py::test_english_and_italian_rule_data` | 1 | **PASS** |

The 24 integration cases are opt-in: ASR and diarization use
`MEETSCRIBE_ASR_TESTS=1`, browser tests use `MEETSCRIBE_BROWSER_TESTS=1`,
the Docker health test uses `MEETSCRIBE_DOCKER_TESTS=1`, and the WeTextProcessing
test uses `MEETSCRIBE_TEXT_NORMALIZATION_TESTS=1`.

### Checkpoint WER

The explicit-30-second checkpoint run used NeMo normalization and JiWER.
`S/D/I` means substitutions, deletions and insertions. The gate requires WER
at most 3%, zero reference-word deletions and no selected-negation differences.

| Recording | WER | S/D/I | Negation differences | Gate |
|---|---:|---:|---:|---|
| English WAV | 1.72% | 7 / 0 / 5 | 0 | PASS |
| English MP3 | 1.44% | 6 / 0 / 4 | 0 | PASS |
| Italian WAV | 0.68% | 1 / 0 / 0 | 0 | PASS |
| Italian MP3 | 0.00% | 0 / 0 / 0 | 0 | PASS |

These scores match the earlier 30-second default comparison. The table measures
word errors after normalization, not semantic correctness; for example, the
English output contains `Elena` → `Elina`, and Italian WAV contains `riletta`
→ `diletta`. See [the window comparison](../investigations/asr-window-comparison.md) for the
25-, 28- and 30-second results and limitations.

### Two-voice transcript failure

The exact-match test reported `Friday` → `Fragi` and `Elena` → `Elina`.
The observed transcript has two substitutions, no deletions and no insertions
under basic normalization: 2 errors in 42 reference words, approximately
4.76% WER. This is a separate diagnostic calculation, **not** a
NeMo-normalized checkpoint score or the test's pass criterion. The test
currently fails on any normalized text difference.

The 20.18-second recording fit in one 30-second model window and VAD retained
it in full. Whisper transcribed the isolated Flite clip as `Friday`, but the
same clip inside the two-voice recording as `Fragi`. This suggests sensitivity
to context; it does not prove the exact cause of the substitution.

## Historical reports and reruns

Earlier results and their environments are documented in the
[Sprint 1 checkpoint](../reports/sprint-1-checkpoint.md) and
[diarization feasibility report](../reports/sprint-2-diarization-feasibility.md).
They may use different configurations and should not be treated as results
from the targeted runs above.

The real-model tests require provisioned local models, Docker and appropriate
GPU resources. For example, to rerun the checkpoint:

```sh
MEETSCRIBE_ASR_TESTS=1 uv run --locked python -m pytest -q -s \
  test/integration/test_checkpoint_wer.py
```

## Experimental branch weekday normalization (2026-11-22)

The experimental extractor on `llm-normalize-python-summary` normalizes an
explicit weekday typo only when one weekday is a strong, unique match and
named-term checks do not apply. The corrected phrase appears in the action
deadline and summary; the original transcript remains unchanged, and no
calendar date is inferred. On that branch, the unit suite passed **308 tests**
in **14.79 seconds** with one existing Starlette/AnyIO deprecation warning.
The filtered real two-voice quality test passed **1 integration test** in
**77.59 seconds**, verifying `on Friday` while the ASR transcript still says
`Fragi`. Unit tests also cover an independent English typo, an Italian weekday
typo, weak matches and named-term counterexamples. `main` contains the quality
checks but not this implementation, so they may fail there until the branch is
merged. The full integration suite and 297-second context test remain unrun;
this run does not establish improved ASR accuracy.

Update this snapshot after relevant test runs, recording skipped and unrun
checks separately from failures.

## Large-v3 model and quality rerun (2026-11-23)

The app now defaults to full `Systran/faster-whisper-large-v3`, with model
directory and artifact manifest configurable as a pair; large-v3-turbo remains
available. In the earlier one-clip, 20.2-second synthetic comparison, Turbo
produced `Fragi` and `Elina` (2/42 word errors; 4.76% WER), while full
large-v3 produced `Friday` and `Elena` (0/42).

| Model | Transcription time | Change vs. Turbo | Sampled peak GPU memory |
|---|---:|---:|---:|
| large-v3-turbo | 0.774 s | baseline | 1,238 MiB |
| full large-v3 | 1.435 s | +0.661 s (+85.4%) | 2,294 MiB |

For this standalone inference test, full large-v3 was **+85.4%** slower than
Turbo. This is one 20.2-second synthetic clip, not a representative
performance average.

The real-model ASR and speaker-quality command passed **8 tests and reported
1 expected XFAIL** in **201.74 seconds**. The four checkpoint recordings met
the WER ≤ 3%, zero-deletion and selected-negation gates: English WAV/MP3 each
scored 1.29% WER, Italian WAV/MP3 each scored 0.00%, with no deletions or
selected-negation differences. The known repeated-sentence omission remained
an XFAIL. Quiet negations/silence, five-minute timestamps, diarization and
speaker alignment, including the two-voice exact-text check, passed. On this
checkpoint, `Elena` was still transcribed as `Elina` twice; the quality gates
do not rule out substitutions.

| Quality check | Result | Elapsed time |
|---|---|---:|
| Real-model ASR and speaker-quality batch | 8 passed, 1 expected XFAIL | 201.74 s |
| Isolated two-voice exact-text integration check (earlier targeted run) | PASS | 43.62 s, including image build |
| Full three-case audio-to-minutes run | Failed on English context budget | 61.33 s |
| Italian plus known-error minutes selection | Failed on Italian context budget; known-error case not reached | 34.18 s |
| Isolated known-error minutes case | Failed quality assertion on project-status decision | 85.49 s |
| Weekday-deadline assertion unit tests | 8 passed | 0.09 s |

### Side-by-side timing for the ASR/speaker test batch

| Test | large-v3-turbo (2026-11-21) | full large-v3 (2026-11-23) |
|---|---:|---:|
| Synthetic WAV/MP3 accuracy | 5.16 s | Individual time not captured; included in 201.74 s batch |
| Four-recording checkpoint WER | 60.07 s | Individual time not captured; included in batch |
| Quiet negations and silence | 5.18 s | Individual time not captured; included in batch |
| Five-minute MP3 | 10.95 s | Individual time not captured; included in batch |
| Repeated sentence | 4.74 s | Individual time not captured; included in batch |
| Speaker alignment, shared English/Italian fixture | 31.90 s shared-fixture setup | Individual time not captured; separate exact-text run took 43.62 s including image build |
| Community-1 diarization adapter | 14.32 s | Individual time not captured; included in batch (does not run ASR) |

For the same selected batch, the old Turbo per-test/fixture times sum to
**132.32 s** across **9 test items**; the full-large-v3 batch took **201.74 s**
for those 9 items. That gives approximate average elapsed times of **14.70 s**
per item for Turbo and **22.42 s** for full large-v3: **about +52.5%**. This is
a rough batch-level average, not an average of paired per-test percentages:
the Turbo values are historical per-test/fixture durations from the larger
all-opt-in run, while the new model run recorded only the selected batch total.
The batch also includes one diarization-only test.

For the filtered known-error minutes test, the recorded total was **77.16 s**
with Turbo and **85.49 s** with large-v3 (**+10.8%**). The transcription
stage in the respective two-voice quality runs was **19.581 s** and **23.623 s**
(**+20.6%**); that stage includes the production transcription workflow and is
not the standalone inference timing above. These overlapping measurements are
not added to the batch average.

The known-error case report measured **23.623 s** for transcription and
**57.245 s** for minutes generation. Other per-test durations within the
201.74-second ASR batch were not captured by that pytest invocation; the time
above is the whole batch duration. The run-level times are from separate
invocations and should not be summed; stage timings exclude other setup and
teardown time.

The all-case audio-to-minutes run failed on the English checkpoint before
minutes generation: estimated transcript context was **11,058 tokens versus
4,096**. A separate run reached the Italian checkpoint and failed before
generation at **4,461 versus 4,096 tokens**; the short known-error case was not
reached in that run. Running the known-error case alone completed generation
but failed its quality assertion because “Project is on schedule” was
classified as a decision. The transcript correctly contained `Friday` and
`Elena`; the action deadline was `Friday`. The results-review action was
present and the in-meeting service description was not turned into an action.
The deadline assertion was not reached because the decision assertion failed
first. The decision issue remains a failure; the deadline wording is addressed
by the test-only update below. Thus, none of the three minutes-quality cases
passed in the recorded rerun. No other integration tests were run.

**Minutes deadline assertion update (2026-11-23):** The known-error quality
check now accepts either the exact source phrase (for example, `on Friday` or
`by Friday`) or the weekday alone (`Friday`, case-insensitive). It still
rejects a changed preposition or expanded wording. The focused helper tests
passed **8 tests**; Ruff, formatting and mypy passed. The real-model minutes
test was not rerun after this test-only adjustment; the recorded known-error
run still fails on the separate project-status-as-decision check.

**English WAV minutes-context rerun (2026-11-23):** With a 12,288-token
context and 1,024-token output limit, the selected `english-checkpoint`
integration test passed in **126.09 seconds**. The 297.305-second recording
produced 49 segments; extraction used a **10,290-token prompt upper bound**,
**2,901 runtime prompt tokens** and **812 generated tokens**. Minutes
generation took **67.259 seconds**. Sampled GPU use peaked at **4,444 MiB**
during minutes generation and **2,294 MiB** during transcription, out of
**6,141 MiB** total; no GPU error occurred. This validates generation and
export shape for this fixture, not semantic quality.

**Historical English MP3 minutes rerun:** The selected `english-checkpoint`
integration test uses `meeting-en.mp3` and passed with a 16,384-token context and
4,096-token output limit in **203.09 seconds**. The 297.305-second recording
used a **10,199-token prompt upper bound**, **2,862 runtime prompt tokens** and
**1,023 generated tokens**; minutes generation took **120.283 seconds**.
Sampled GPU use peaked at **4,710 MiB** during minutes generation and **2,294
MiB** during transcription, out of **6,141 MiB** total; no GPU error occurred.
This validates generation and export shape for this fixture, not semantic
quality. Italian and known-ASR-error cases were not included, so the
three-case checkpoint remains incomplete.
