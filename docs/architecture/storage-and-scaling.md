# Storage and team scaling

This is a planning estimate, not a measured capacity or a commitment to a
particular deployment. Persistent meetings, storage quotas, deletion and
multiuser access are not implemented. See [the delivery plan](../../plan.md)
for current limits and implementation slices.

## Intended behavior

- Keep each original uploaded recording with its meeting until manually
  deleted or seven days after upload. The retention period is the current
  local-MVP plan; a company deployment must agree its own policy.
- Allow the original audio to be downloaded while retained, without a browser
  player. Downloaded copies are outside the application's deletion control.
- When a storage quota or reserved-free-space threshold prevents a new
  upload, explain the problem and prompt the user to delete existing
  meetings; do not silently delete unexpired meetings to make room.
- The app is configured to accept at most three hours / 10 GB per recording.
  This user-requested test ceiling is not a measured capacity guarantee; the
  app has not demonstrated three-hour inference or sufficient free space for a
  10 GB upload. The earlier 30-minute / 500 MB local-MVP estimate is superseded
  as a configured limit, not as evidence of tested capacity.

## Current temporary behavior

The current app keeps at most 16 transcription results and draft-minutes
sessions in process memory, expiring completed results after up to one hour.
They are lost on refresh or restart; the uploaded audio is removed after
transcription. This temporary behavior is not the planned seven-day retention
and does not preserve original audio for later download. Persistent meeting
records, saved audio, deletion, expiry and storage quotas remain future work.

## Estimating original audio

These examples assume one two-hour recording per meeting, a constant daily
rate for seven days, no manual deletion and all seven days of recordings
still retained. Decimal units are used: 1 GB = 1,000 MB. Actual sizes depend
on the recording format and encoding; variable-bitrate MP3 sizes vary.

| Encoding | Approximate size per meeting | 1 meeting/day, 7 days | 5 meetings/day, 7 days | 20 meetings/day, 7 days |
|---|---:|---:|---:|---:|
| MP3, 128 kbps | 115 MB | 0.81 GB | 4.03 GB | 16.13 GB |
| PCM WAV, 16 kHz, 16-bit mono | 230 MB | 1.61 GB | 8.06 GB | 32.26 GB |
| PCM WAV, 44.1 kHz, 16-bit stereo | 1,270 MB | 8.89 GB | 44.45 GB | 177.81 GB |

For constant-bitrate MP3, bytes are approximately `bitrate in bits/second *
duration in seconds / 8`. For uncompressed PCM WAV, audio bytes are
approximately `sample rate * (bits per sample / 8) * channels * duration in
seconds`, plus a small header. For example, 128,000 bits/second for 7,200
seconds gives 115,200,000 bytes. Multiply the recording size by meetings per
day and retained days for a steady-rate estimate.

These figures cover **original audio only**. They do not include temporary
uploads, decoded audio or processing buffers, transcripts and summaries,
database/WAL files, logs, model weights, container images, or filesystem
overhead. Some may consume disk, RAM, or both depending on implementation.
Concurrent uploads and processing may temporarily need additional copies or
space. Models and other applications can share the same underlying disk even
when their files are outside the meeting-storage quota.

## Current summary-model artifacts

The pinned Qwen3 4B Q4_K_M model layer is 2,497,280,480 bytes (about 2.50 GB
decimal). The digest-pinned Ollama 0.40.17 image's Linux/amd64 registry layers
total about 3.16 GiB compressed; extracted image storage differs. These are
shared host/VM disk costs, not per-meeting costs, and are separate from the
ASR and diarization models. Artifact sizes do not establish RAM or VRAM use;
that depends on runtime settings and must be measured on the target hardware.

One sequential ASR -> diarization -> Qwen3 -> ASR handoff run on the
development RTX 4050 used a 4,096-token summary context. Total GPU memory
peaked at 3.36 GiB of 6.0 GiB, and host memory use peaked at 11.1 GiB of
15.3 GiB; swap peaked at 5.12 GiB of 8.0 GiB, from a 4.28 GiB baseline. All
stages completed without OOM or GPU offload. This is a single synthetic-audio
measurement, not a capacity guarantee, a two-hour throughput result, or a
prediction for the future VM; see the [slice 3.2 report](../reports/sprint-3-slice-3.2.md).

For illustration, with a hypothetical **50 GB quota for meeting originals
alone**, 20 daily two-hour 44.1 kHz stereo WAV files add about 25.4 GB per
day, reaching the quota during the second day, well before seven-day expiry.
This ignores all overhead and is not a proposed quota. Available host free
space may become the binding limit even before an application quota.

## Handling capacity

The planned local storage uses persistent Docker-backed storage on the host
or VM; the actual volume, filesystem, location and capacity are not yet
specified. A Docker volume is not automatically infinite and may share disk
space with model weights, images and unrelated workloads.

Before accepting work, account for the configured meeting-data quota,
available filesystem space, required free-space reserve, outstanding uploads
and estimated temporary processing space. Concurrent admissions must not
reserve the same space twice. If enough space cannot be reserved, reject the
upload with an actionable message, let the user review and delete meetings,
and allow a retry after cleanup. Keep review and deletion available even when
storage is full. Surface deletion or cleanup failures rather than claiming
space was freed. Do not select or delete unexpired meetings automatically.

The seven-day timer is measured from upload, not the last edit. Access must
be denied at expiry even if scheduled cleanup has not run yet; physical
cleanup happens when the app is running and can fail or need retrying.
Manually deleting or expiring a meeting removes application-managed copies,
not files already downloaded or host backups; forensic erasure is not
guaranteed.

## Team use and processing capacity

More users can increase both stored data and upload/processing demand. The
table uses total meetings per day across the team, not meetings per person.
For example, five users uploading one meeting each per day correspond to the
five-meetings/day column. Actual overlap between meetings, upload bursts,
working days and retention timing affect peak storage.

Storage capacity alone does not determine throughput. The current design
runs one inference job at a time and serializes GPU-heavy stages. If a
two-hour recording takes `T` hours end-to-end to process, five such
recordings arriving per day require roughly `5 * T` processing hours per
day, before retries and overhead. If demand exceeds available processing
time over a sustained period, the queue and wait time grow. No reliable
value for `T` is known for two-hour meetings, and the future VM's GPU, CPU,
RAM, disk and network capabilities are unknown.

The current app is a local single-user development system, not a team
service. Supporting a team also needs a deliberate design for:

- Authentication, authorization and meeting ownership/sharing, including
  who can download or delete each meeting. A user must not be prompted to
  delete data they are not authorized to manage.
- Simultaneous uploads, bounded queue length and storage reservations,
  fairness, job status, cancellation and recovery after interruption.
- Measured inference throughput, latency, memory and disk usage on the
  actual VM, plus safe worker ownership and capacity for peak demand.
- Company-approved network exposure, transport security, storage,
  operational retention, backups and monitoring. Docker portability alone
  does not provide these controls.

Before selecting a quota or scaling architecture, establish the target VM
resources, expected users and daily/peak recording volume, formats and sizes,
retention rules, and acceptable processing delay. Measure representative
end-to-end jobs before deciding whether one worker suffices or additional
resources are required.
