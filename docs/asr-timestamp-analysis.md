# Analysis: out-of-bounds ASR timestamps

## Symptom

The first-release checkpoint used a user-reviewed, synthetic English MP3,
297.305 seconds long. The pinned faster-whisper 1.2.1 model returned segments
beyond the recording duration. One run produced timestamps up to 326.02 seconds;
`TranscriptionService` rejected the result instead of clamping timestamps.

The matching WAV and the synthetic Italian WAV and MP3 did not exhibit this
timestamp failure in the initial check. All files passed upload validation.

## Investigation

Runs used the provisioned model in a GPU container with networking disabled.
The adapter used integrated VAD, beam size 5 and the faster-whisper default
`condition_on_previous_text=True`. In one diagnostic MP3 run, the model repeated
"Thank you" after the last spoken sentence, with six segments ending past the
recording. Another run produced a long repeated phrase within a segment ending
at 297.300 seconds: that transcript passed timestamp validation but was still
incorrect. Output varies between runs, so a valid timestamp alone does not prove
the absence of hallucination.

Changing the input to WAV, disabling VAD, or disabling previous-text conditioning
each avoided out-of-bounds segments in the respective diagnostic runs. These are
comparisons, not proof that VAD timestamp restoration or MP3 decoding caused the
failure. The behavior is consistent with a model repetition loop when previous
text is reused, but the exact internal cause has not been established.

## Change and verification

The adapter now explicitly sets `condition_on_previous_text=False` while retaining
the same integrated VAD, model, and beam size. This reduces exposure to previous
text repetition, at the potential cost of less consistent cross-window context.
The selected setting is recorded in transcript provenance. The service still
rejects invalid timestamps; it does not clamp, discard, or silently retry them.

On the development GPU, the opt-in regression test transcribed the English MP3
offline and checked that timestamps fit within the recording, the final expected
sentence was present, and the observed repetition was absent. The existing
English and Italian quiet-negation GPU smoke also passed. A separate offline run
produced valid transcripts for both English and Italian WAV and MP3 recordings.
In that run, the English MP3 ended at 296.94 seconds, within 297.305 seconds.

Compared with the generated script, an exploratory token comparison found
approximately 233 differing tokens in one previous-settings MP3 output with
in-bounds repetitive text, versus 14 with conditioning disabled. This is not
WER: it uses a simple sequence matcher, does not normalize spoken numbers, and
the reference is not fully human-annotated. The generated voice may also speak
"colon" in one sentence. Do not treat these figures as a quality threshold.

The synthetic eSpeak NG recording is stored at `test/data/meeting-en.mp3`
(SHA-256 `ef3d4e68a1be6e18fc08788d08e363e3b75f6246a510ae488f10543dde4db69c`).
To rerun the specific regression:

```sh
MEETSCRIBE_ASR_TESTS=1 \
  uv run --locked python -m pytest -q test/integration/test_asr_long_recording.py
```

This setting does not guarantee valid timestamps or accurate words for all inputs. First-release acceptance
still requires the planned human timing review, mismatch analysis, cancellation
check, and representative recordings.
