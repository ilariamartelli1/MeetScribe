# Test status

Snapshot: 2026-11-21, based on repository commit `e9fbd6b`. This document
records observed results, not a live dashboard. Results apply to the stated
environment and test inputs; reruns may differ.

**Slice 2.3a update:** After this snapshot, browser editing was removed and the
targeted browser-flow and browser-asset checks passed: 12 tests total, including
8 browser cases and 4 asset checks. The browser test previously named
`test_real_api_upload_edit_and_download` was replaced by
`test_real_api_upload_read_only_and_download`, and a multiple-segment download
case was added. The historical snapshot counts below have not been rerun as a
full suite with these changes; the other integration results remain earlier
observations.

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
→ `diletta`. See [the window comparison](asr-window-comparison.md) for the
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
[Sprint 1 checkpoint](sprint-1-checkpoint.md) and
[diarization feasibility report](sprint-2-diarization-feasibility.md).
They may use different configurations and should not be treated as results
from the targeted runs above.

The real-model tests require provisioned local models, Docker and appropriate
GPU resources. For example, to rerun the checkpoint:

```sh
MEETSCRIBE_ASR_TESTS=1 uv run --locked python -m pytest -q -s \
  test/integration/test_checkpoint_wer.py
```

Update this snapshot after relevant test runs, recording skipped and unrun
checks separately from failures.
