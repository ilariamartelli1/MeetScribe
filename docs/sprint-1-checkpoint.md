# Sprint 1 checkpoint: preliminary evaluation

Date: 2026-11-11. This is not first-release acceptance: the synthetic reference
text and generated timing boundaries still need human verification, and the
recordings are not representative of a range of real speakers or meetings.
Only fictional, synthetic audio was used.

For the separate experiment with previous-text conditioning enabled, see
[the comparison report](sprint-1-conditioning-enabled.md).

## Runs and limitations

- The English recording is 297.305 seconds; the Italian recording is 73.318
  seconds. Each was transcribed in WAV and MP3 through the browser, Docker app,
  spawned worker, and provisioned faster-whisper model on the RTX 4050.
- Separate runs of all four files used the production ASR adapter inside a
  container with networking disabled. Both WAV/MP3 pairs matched the hashes
  in their draft reference JSON files.
- The English reference has 48 generated sentence intervals and the Italian
  reference has 12. Each interval overlapped at least one ASR segment in each
  offline run. All returned segments were within recording bounds. The 11
  English and four Italian generated quiet-word intervals each overlapped a
  segment containing that word. This is not word-level alignment, a count of
  individual repeated words, or proof that the generated boundaries are correct.
- The generated reference includes names, numbers, amounts, dates and speech
  following pauses. It uses synthetic single-speaker speech; further coverage
  and human annotation are needed before calling it representative.

## Raw transcript differences

The table compares the four browser-run outputs to the draft scripts.
Matching ignores case and punctuation; it does not normalize numbers or
currency symbols. Times are approximate **draft sentence** intervals on the
original recording, not human-verified word timestamps. The descriptions below
inspect the full text, including currency symbols. Differences may reflect the
generated speech or unverified reference rather than model mistakes. The
`Captured runs` column counts the separate three-run adapter comparison below;
the browser run is not included in that count.

| Audio | Draft interval | Type | Reference and observed output | Captured runs |
|---|---|---|---|---|
| EN WAV and MP3 | 00:13.681–00:18.950 | Name | `Elena` → `Elina` | 3/3 each |
| EN WAV and MP3 | 00:25.241–00:31.813 | Format spelling | `wave` → `WAV` | 3/3 each |
| EN WAV | 00:32.163–00:38.077 | Spelling/spacing | `timestamps` → `time stamps` | 3/3 |
| EN MP3 | 00:32.163–00:38.077 | Article substitution | `a reviewer` → `the reviewer` | 3/3 |
| EN WAV and MP3 | 01:02.560–01:07.912 | Date formatting | `fourteen April` → `14 April` | 3/3 each |
| EN WAV and MP3 | 01:14.902–01:20.340 | Amount formatting | `two thousand four hundred euros` → `€2,400` | 3/3 each |
| EN MP3 | 02:18.740–02:24.889 | Name | `Elena` → `Elina` | 3/3 |
| EN WAV and MP3 | 03:51.266–03:57.008 | Date formatting | `twenty seven May` → `27 May` | 3/3 each |
| EN WAV and MP3 | 03:57.358–04:04.115 | Number formatting | `fifteen` → `15` | 3/3 each |
| IT WAV and MP3 | 00:29.685–00:36.541 | Date and amount formatting | `quattordici` → `14`; `duemila quattrocento` → `2400` | 3/3 each |
| IT WAV | 00:55.394–01:00.769 | Word substitution | `riletta` → `diletta` | 3/3 |
| IT WAV and MP3 | 01:07.704–01:12.968 | Date formatting | `ventisette` → `27` | 3/3 each |

Number/date formatting and the currency symbol preserve the apparent value;
they should not automatically be counted as recognition errors. The English
voice says "colon" aloud, so its transcription is not a discrepancy. These
comparisons are not WER and do not establish transcription quality on other
recordings.

The browser allowed correction of `Elina` to `Elena` for both English files
and `diletta` to `riletta` for the Italian WAV; the downloaded draft contained
the corrections and timestamps. The Italian MP3 did not contain those specific
differences in this run; its unmodified draft was downloaded and checked.

### WAV and MP3 differences

The WAV and MP3 versions contain the same synthetic speech, but decoding the
lossy MP3 does not reproduce the WAV's audio samples exactly. Differences in
quiet sounds and transitions can affect speech detection, segment boundaries
and the model's choice of words.

The runs above showed some format-specific differences, such as `riletta` →
`diletta` in the Italian WAV but not the MP3. We have not established why that
specific difference occurred. Model outputs can also vary between runs, so one
comparison cannot attribute an error solely to the audio format.

### Three captured runs with conditioning disabled

The original offline adapter capture was followed by two more captures of each
recording, using the same model, settings, GPU and fixtures. In all three
captures, the segment texts and timestamps were identical for each recording;
all 12 transcripts passed service validation. Each generated quiet-word
interval overlapped a segment containing that word, which does not establish
word-level timing or count repeated words.

The table's `3/3 each` means the listed output appeared in all three captures
for **each** format named in that row. The extra captures were adapter runs,
not browser runs; the timing references still need human review.
The original full test suite and the two repeated full suites each passed all
116 tests. Three runs on the same synthetic fixtures are not a representative
quality benchmark and do not guarantee repeatability on other audio.

## Cancellation and other checks

A browser cancellation was performed on the English MP3 after the real worker
was observed allocating 1,188 MiB of GPU memory. The worker PID disappeared
from both the container process list and the GPU process list; total reported
GPU usage returned from active use to its 31 MiB baseline. A subsequent real
transcription succeeded and its edited TXT downloaded. GPU memory was observed
on the host, not attributed to the model alone. The test did not instrument the
model's call stack, so it does not prove which inference phase was interrupted.

The default test suite, Ruff, formatting, mypy, opt-in browser and CPU Docker
tests, and the documented offline GPU tests passed. Local and isolated GPU
Compose health endpoints responded; the GPU container saw the NVIDIA device.
The existing model and dependencies were reused: the README's fresh dependency
installation and model download commands were not rerun. The browser Compose
run did not block network access; offline operation was checked separately as
described above.

## Human review still required

Listen to the two recordings against
`test/data/checkpoint/english/meeting-en-script.txt` and
`test/data/checkpoint/italian/meeting-it-script.txt`, correct the spoken text
and verify the generated sentence and quiet-word boundaries in the
`meeting-*-draft-reference.json` files in those same directories. In particular,
check the discrepancies above, quiet negations, names, amounts, dates and speech
after pauses. Review both formats for encoding-related differences. Until this
review and the representativeness assessment are complete, slice 1.5 remains open.
