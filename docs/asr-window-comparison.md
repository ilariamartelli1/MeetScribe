# Analysis: 25-, 28- and 30-second ASR windows

Initial comparison: 2026-11-19. Three-way comparison: 2026-11-20.

## Conclusion

The model's default 30-second window achieved the lowest WER and no aligned
deletions on all four synthetic checkpoint recordings. Both the 25- and
28-second windows produced missing or incorrect words. At 28 seconds, all four
recordings failed the zero-deletion gate, although each stayed below 3% WER.
These results support returning to the 30-second default for this project,
subject to rerunning the related regression checks.

The comparison does not establish that 30 seconds is always better, or explain
the exact internal decoding mechanism behind each error. All settings can
produce transcription mistakes.

## Background

The adapter had been changed experimentally to pass `chunk_length=25` to
faster-whisper. The experiment followed an observation that the model sometimes
omitted a repeated sentence in a synthetic two-voice recording. The 25-second
setting did not reliably preserve that repetition.

The checkpoint quality gate also changed: it now requires zero JiWER
reference-word deletions in addition to the existing maximum of 3% WER per file
and no differences in selected negation words. A file can therefore fail the
new gate while still passing the previous WER threshold.

## Method

The comparison used the same current app image, model, GPU, reference texts and
four versioned synthetic recordings:

- `test/data/checkpoint/english/meeting-en.wav` and `meeting-en.mp3`
- `test/data/checkpoint/italian/meeting-it.wav` and `meeting-it.mp3`

The recordings matched the SHA-256 hashes in their draft reference JSON files.
The reference scripts were reviewed for spoken text; their generated timing
boundaries have not been independently verified.

All settings used the production `FasterWhisperTranscriber` and
`TranscriptionService` in offline Docker containers on an NVIDIA RTX 4050
Laptop GPU. Shared settings included:

- `faster-whisper` 1.2.1, CTranslate2 4.8.5 and PyAV 18.1.0.
- `deepdml/faster-whisper-large-v3-turbo-ct2`, revision
  `4df90f75321148c3a29a9e2351b7ddf8f5b115a8`, with CUDA `int8_float16`.
- Integrated VAD enabled, beam size 5, and
  `condition_on_previous_text=False`.
- Explicit English or Italian language selection, NeMo text normalization and
  JiWER scoring against the same reviewed reference.

For the comparison, the adapter's chunk-length constant was overridden in the
container process without changing repository files. `None` used the provisioned
model's 30-second feature-extractor default; `28` and `25` selected those
window lengths. Each setting used a fresh container and model instance, avoiding
feature-extractor state carrying over between settings. The model weights were
not modified.

The 2026-11-20 run re-tested all four files at each setting in the same app
image. Its 25- and 30-second WER and error counts matched the earlier
comparison. Two earlier 30-second runs also returned identical segments and
scores. The 28-second results below come from one four-file run; a separate
instrumented run reproduced its English MP3 and Italian WAV transcripts and
recorded window boundaries. These few runs do not establish statistical
reliability.

## Results

`S`, `D` and `I` are JiWER substitutions, deletions and insertions after
normalization.

| Recording | Reference words | 30 s WER | 30 s S/D/I | 28 s WER | 28 s S/D/I | 25 s WER | 25 s S/D/I |
|---|---:|---:|---:|---:|---:|---:|---:|
| English WAV | 696 | 1.72% | 7 / 0 / 5 | 2.30% | 9 / 1 / 6 | 2.01% | 9 / 0 / 5 |
| English MP3 | 696 | 1.44% | 6 / 0 / 4 | 2.30% | 11 / 1 / 4 | 2.59% | 11 / 1 / 6 |
| Italian WAV | 148 | 0.68% | 1 / 0 / 0 | 2.70% | 1 / 3 / 0 | 3.38% | 3 / 2 / 0 |
| Italian MP3 | 148 | 0.00% | 0 / 0 / 0 | 2.70% | 1 / 3 / 0 | 3.38% | 3 / 2 / 0 |

No selected-negation differences were reported for any setting. Under the
current gate, the 30-second scores pass for all four recordings. At 28 seconds,
all four fail the zero-deletion requirement: English WAV and MP3 each have one
aligned deletion, and Italian WAV and MP3 each have three. At 25 seconds, the
English MP3 fails the zero-deletion requirement; both Italian recordings fail
the zero-deletion and 3% WER requirements. The 25-second English WAV passes
those gates despite the meaning-changing error below.

## Examples of differences

### English budget

The source reference says the estimated budget is **two thousand four hundred
euros**. The 30-second output rendered it as `€2,400`.

The 25-second WAV output rendered the relevant passage as:

> The estimated preparation budget is $2,000 for her. euros for the first review.

The 25-second MP3 output likewise included `$2,000` followed by `euros`. This
changes both the amount and the currency representation. It illustrates why
passing a WER threshold or reporting zero aligned deletions does not establish
that a transcript preserves important facts. At 28 seconds, both English files
rendered the amount as `2400 euros` or `2,400 euros`.

### Italian omitted speech

The reference contains:

> I nomi e le cifre in questa registrazione sono inventati per il test.

The 30-second WAV output retained that sentence. The 25-second WAV output was:

> I nomi e le cifre in questa registrazione sono in me.

JiWER reported `inventati per` as substitutions and `il test` as deletions.
The 25-second MP3 produced the same differences. At 28 seconds, both Italian
files retained the complete sentence.

### A different Italian omission at 28 seconds

The reference contains:

> La bozza non è un verbale verificato e deve essere riletta da una persona.

At 28 seconds, both Italian files instead ended the sentence with:

> La bozza non è un verbale verificato e deve essere veri.

JiWER aligned `riletta` as a substitution and `da una persona` as three
deletions. The 30-second WAV retained `da una persona` but rendered `riletta`
as `diletta`; the 30-second MP3 retained the complete reference sentence.
The 28-second English MP3 also had an aligned deletion of `date` in the
phrase `an incorrect date`.

### Window boundaries

Faster-whisper does not apply a special overlap rule for the 30-second setting.
Its next window position depends on the timestamp tokens predicted by Whisper.
When the output has a single ending timestamp, or no consecutive timestamp
tokens, faster-whisper advances by the full selected window size. When the
output has consecutive timestamps and an unfinished final segment, it instead
advances to the last completed timestamp; if that position is before the
window's end, the next input window overlaps. This describes the normal
timestamp-based path used here, not every possible faster-whisper option.

Instrumented runs captured actual model input windows for the English MP3 and
Italian WAV. In the English MP3 traces:

- At 30 seconds, the window at 0–30 seconds was followed by one starting at
  23.12 seconds, an overlap of 6.88 seconds.
- At 28 seconds, the window at 56–84 seconds was followed by one starting at
  78.44 seconds, an overlap of 5.56 seconds.
- At 25 seconds, the observed windows advanced by 25 seconds without overlap.

These observations are specific to the recordings and runs tested. **A
25-second window can overlap on another recording** if the model predicts
timestamps that make faster-whisper restart before the window's end. Changing
window length changes the audio boundaries and context seen by the model, which
can change its predicted timestamps; the trace does not establish exactly why
any particular timestamp was predicted.

At 28 seconds, the Italian windows covered 0–28, 28–56 and
56–69.94 seconds on the VAD-compacted timeline, with no gaps or overlaps.
The second boundary mapped to approximately 59.38 seconds in the original
recording, inside the sentence ending in `riletta da una persona`; the
generated sentence interval is 55.39–60.77 seconds and has not been
independently verified. The association is consistent with a boundary-related
error, but does not prove its internal cause.

Neither traced 28-second file showed a gap between input windows. All offsets
above are on the VAD-compacted timeline unless identified as original
recording times. VAD removes audio it classifies as silence before windows are
selected, so compacted offsets differ from original recording timestamps. This
does not prove that other recordings cannot have gaps or VAD omissions.

### Errors that remain at 30 seconds

The 30-second output still transcribed `Elena` as `Elina` in the English
recording, and the Italian WAV output contained `riletta` as `diletta`.
Restoring the default is therefore a mitigation for the observed regression,
not a guarantee of correct names, facts or wording.

## Interpretation and limitations

Changing the window changes which audio context and segment boundaries the
decoder processes. The traced boundaries support that as a plausible
explanation for some errors, but do not reveal the model's internal decisions
or prove the exact causes. The zero-deletion gate changes the pass/fail decision
but does not cause the observed text or WER differences.

An earlier checkpoint documented an English WAV WER of 1.44%, compared with
1.72% for the current 30-second comparison. The app environment changed since
that checkpoint, including CUDA library versions and diarization dependencies.
The current side-by-side comparison controls the environment more closely than
the historical comparison; it does not isolate the effect of those earlier
environment changes. The earlier results are in the
[Sprint 1 checkpoint](sprint-1-checkpoint.md); a later 30-second run on the
diarization-enabled stack is documented in the
[diarization feasibility report](sprint-2-diarization-feasibility.md).

The recordings use synthetic, single-speaker English and Italian voices. WER
depends on normalization and token alignment; an aligned deletion is not always
equivalent to a manually verified acoustic omission, and important substitutions
can occur without a deletion. These results do not measure performance on real
meetings, accents, overlapping speech or other languages.

## Recommended next steps

1. Use an explicit 30-second window and preserve the stricter omission checks.
   The adapter now explicitly passes `chunk_length=30`; the comparisons above
   used a fresh model's 30-second default rather than an explicit value.
2. Rerun the checkpoint WER, long-recording timestamp, quiet-negation and
   speaker-alignment checks on the restored setting. Separately record the
   repeated-sentence diagnostic as an expected omission where applicable.
3. Review names, numbers, dates, negations, decisions and action items against
   authorized or synthetic audio. Do not treat aggregate WER as a guarantee of
   summary accuracy.

The current adapter configuration can be checked with:

```sh
MEETSCRIBE_ASR_TESTS=1 uv run --locked python -m pytest -q -s \
  test/integration/test_checkpoint_wer.py
```

This command uses the adapter's current setting; it does not independently
compare the three window lengths.
