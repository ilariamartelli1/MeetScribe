# Sprint 1 checkpoint: preliminary evaluation

Date: 2026-11-11. This is not first-release acceptance: the synthetic reference
text and generated timing boundaries still need human verification, and the
recordings are not representative of a range of real speakers or meetings.
Only fictional, synthetic audio was used.

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

The table compares the four browser-run outputs to the generated scripts as
they stood during the run, before the confirmed spoken "colon" was added.
Matching ignores case and punctuation; it does not normalize numbers or
currency symbols. Times are approximate **draft sentence** intervals on the
original recording, not human-verified word timestamps. The descriptions below
inspect the full text, including currency symbols. Differences may reflect the
generated speech or unverified reference rather than model mistakes.

| Audio | Draft interval | Type | Reference and observed output |
|---|---|---|---|
| EN WAV and MP3 | 00:13.681–00:18.950 | Name | `Elena` → `Elina` |
| EN WAV and MP3 | 00:25.241–00:31.813 | Format spelling | `wave` → `WAV` |
| EN WAV | 00:32.163–00:38.077 | Spelling/spacing | `timestamps` → `time stamps` |
| EN MP3 | 00:32.163–00:38.077 | Article substitution | `a reviewer` → `the reviewer` |
| EN WAV and MP3 | 01:02.560–01:07.912 | Date formatting | `fourteen April` → `14 April` |
| EN WAV and MP3 | 01:14.902–01:20.340 | Amount formatting | `two thousand four hundred euros` → `€2,400` |
| EN WAV and MP3 | 01:33.300–01:39.338 | Reference correction | `no: speaker` → `no, colon, speaker`; the user confirmed that "colon" is spoken |
| EN MP3 | 02:18.740–02:24.889 | Name | `Elena` → `Elina` |
| EN WAV and MP3 | 03:51.266–03:57.008 | Date formatting | `twenty seven May` → `27 May` |
| EN WAV and MP3 | 03:57.358–04:04.115 | Number formatting | `fifteen` → `15` |
| IT WAV and MP3 | 00:29.685–00:36.541 | Date and amount formatting | `quattordici` → `14`; `duemila quattrocento` → `2400` |
| IT WAV | 00:55.394–01:00.769 | Word substitution | `riletta` → `diletta` |
| IT WAV and MP3 | 01:07.704–01:12.968 | Date formatting | `ventisette` → `27` |

Number/date formatting and the currency symbol preserve the apparent value;
they should not automatically be counted as recognition errors. The English
voice says "colon" aloud, so its transcription is correct at that point and
the draft reference has been updated. These comparisons are not WER and do not
establish transcription quality on other recordings.

The browser allowed correction of `Elina` to `Elena` for both English files
and `diletta` to `riletta` for the Italian WAV; the downloaded draft contained
the corrections and timestamps. The Italian MP3 did not contain those specific
differences in this run; its unmodified draft was downloaded and checked.

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

Listen to the two recordings against `test/data/meeting-en-script.txt` and
`test/data/meeting-it-script.txt`, correct the spoken text and verify the
generated sentence and quiet-word boundaries in the corresponding
`meeting-*-draft-reference.json` files. In particular, check the discrepancies
above, quiet negations, names, amounts, dates and speech after pauses. Review
both formats for encoding-related differences. Until this review and the
representativeness assessment are complete, slice 1.5 remains open.
