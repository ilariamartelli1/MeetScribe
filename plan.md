# MeetScribe MVP implementation plan

## Goal and boundaries

Build a Dockerized, single-user, localhost-only application that turns an authorized, non-sensitive test meeting recording into a timestamped, anonymously speaker-labeled transcript and editable meeting minutes. Minutes contain topics, summary, decisions, action items, owners and deadlines; absent owners or deadlines remain "not specified." The operator can listen, correct, mark a draft reviewed, and export text or JSON. Exporting an unreviewed result is allowed, but the export must conspicuously say "DRAFT / UNREVIEWED."

This plan covers the MVP only. It runs locally on the RTX 4050 Laptop GPU (6 GB VRAM, ~15 GB system RAM) before any VM deployment. No company recordings on a third-party VPS; in the MVP use only synthetic or explicitly authorized non-sensitive test recordings. No remote inference: model weights may be downloaded during setup, but audio and transcripts never go to third-party inference services. VM deployment, authentication/multiple users, remote access, live meetings, automatic Teams/Graph integration, and manual Teams transcript import are out of scope. Preserve a documented migration path to a company-controlled VM without claiming the local MVP is production-secure.

## Proposed architecture and operating rules

- Python/FastAPI serves a versioned JSON API (`/api/v1`), a small English-language HTML/CSS/JavaScript UI, and a single sequential processing worker. Business models and orchestration are independent of frameworks/model SDKs; typed interfaces cover speech detection, ASR, diarization, minutes generation, and persistence. Adapters implement these interfaces and can be selected by configuration. Do not invent interfaces without a concrete use; shared schemas (e.g., Pydantic) define API and export behavior.
- Docker Compose starts the app and an internal-only Ollama service. Only the app binds to `127.0.0.1`; Ollama does not publish a host port. Use persistent local volumes for SQLite, audio, and downloaded model weights, with transient working files kept separate. Pin dependencies/images/models and document their licenses/attribution. Do not commit weights, recordings, SQLite files, local configuration, tokens, or IDE files.
- Pipeline: validate WAV/MP3 and declared language (English default; Italian selectable) -> inspect duration (maximum 30 minutes) and upload size (500 MB default, configurable) -> decode/resample for inference -> Silero **VAD** (voice activity detection, not "PAD"), conservatively retain speech regions and map timestamps to the original recording -> faster-whisper `large-v3-turbo` with INT8 GPU inference -> pyannote Community-1 diarization -> align ASR segments to speaker turns and flag overlap/uncertainty -> locally generate structured minutes with a quantized ~4B multilingual Qwen3 model via Ollama -> operator review and export. Verify the exact model and runtime pairing on this machine rather than assuming it fits.
- Default processing mode explicitly requires an available NVIDIA GPU; missing CUDA/model prerequisites produce an actionable error, not an automatic CPU fallback. Explicit CPU mode supports tests and compatible future hardware. Run heavy GPU models sequentially, with one recording processed at a time; unload the ASR model before loading the summary model, bound prompt context, and measure peak VRAM/RAM. Start diarization on CPU to avoid concurrent GPU contention. Never trim silence in a way that loses original timestamps or feeds shortened audio to diarization without remapping.
- SQLite stores meeting metadata, status, timestamps, transcript segments, generated/revised minutes, source references, and review/stale state. The original audio is kept for verification and can be replayed until the meeting is deleted. Persist drafts across restarts. Automatically delete audio, drafts, and associated files seven days after **upload** (not last edit); allow immediate manual deletion. Retention and deletion operate on both database records and files, including failed/interrupted jobs. Never log recording contents or raw transcripts.
- Treat the LLM as an untrusted draft author: require schema-valid JSON; source-segment IDs for decisions/actions; no invented owner/date; preserve an explicit "not specified" value; show an error or review-needed state when generation cannot be validated. Transcript edits mark existing minutes stale; regenerate only on explicit request and do not silently overwrite manually edited minutes. Keep draft/review status unambiguous in the UI and both export formats.

## Sprints, tests, and checkpoints

Every code change gets a relevant automated test in the same commit, and commits stay small enough to describe one tested behavior. Each sprint closes with passing targeted unit/API/UI tests, a working incremental checkpoint, and a commit or series of commits reflecting the verified behavior. CI uses fakes/small fixtures and does not download heavyweight models; local hardware/integration checks are separate and documented.

### Sprint 0 — Repository foundation and runnable shell

- Add a Python package/project layout (`app/domain`, `app/services`, `app/adapters`, `app/api`, `app/static`, `tests`), a configuration module, dependency locking, a minimal FastAPI app, health endpoint, logging/error conventions, `.gitignore`, and a README. Keep code, comments, and developer documentation in English.
- Create Dockerfile and Docker Compose with localhost binding, persistent volumes, NVIDIA runtime instructions/preflight, CPU test mode, and a private Ollama network. Document initial model downloads, Hugging Face consent/token handling for pyannote, offline inference, and model licenses. Add CI for pytest, Ruff, type-checking, and a container smoke check where the runner permits it; use lightweight fake adapters for CI.
- **Checkpoint:** clean clone can run unit tests, boot the Docker app with fakes/CPU mode, and receive a health response. Commit the tested scaffold before adding processing features.

### Sprint 1 — Upload, persistence, lifecycle

- Implement meeting create/list/detail/delete endpoints; accept only WAV/MP3, validate file contents and duration as well as extension/MIME, enforce the 500 MB and 30-minute caps, reject empty/corrupt/no-audio files with clear errors. Store original audio outside the web root; never serve arbitrary filesystem paths. Add SQLite schema/migrations and restart-safe job status (queued, processing, complete, failed), sequential scheduling, explicit failure reporting, cancellation/deletion safeguards, and stale in-progress recovery after restart.
- Add basic upload/list/status/audio-playback UI. Implement retention cleanup at startup and on a scheduled interval, including failed jobs and partial files. Use UTC upload timestamps and protect against path traversal and leaking contents in logs.
- **Tests/checkpoint:** API validation edge cases, persistence across restart, single-job serialization, interruption/recovery, delete and seven-day expiry (including filesystem cleanup); Docker UI upload and playback on a short synthetic clip. Commit each independently verified slice.

### Sprint 2 — Accurate, timestamped local transcription

- Implement the VAD and ASR adapters with configuration-selected models; normalize audio, detect speech conservatively, handle empty speech explicitly, and map every output segment back to original audio timestamps. Use faster-whisper `large-v3-turbo`, INT8-compatible CUDA mode, non-batched inference initially, and a bounded memory footprint. Keep inference local after weight setup; expose progress and actionable model/CUDA errors.
- **Tests/checkpoint:** mock-model tests for VAD boundaries, timestamp remapping, silence, invalid inputs, ASR errors, and language selection; an actual short authorized English and Italian smoke recording on the laptop, including playback/segment alignment and memory logging without content. The UI now shows a timestamped, editable transcript, even before speaker labeling.

### Sprint 3 — Anonymous speaker diarization

- Integrate pyannote Community-1 after accepting its Hugging Face terms and downloading it to a local model volume. Use exclusive diarization for primary ASR alignment while retaining evidence of overlapping/uncertain turns. Assign stable "Voice 1", "Voice 2", ... labels per meeting; do not infer names from the audio. Allow manual correction of segment text and speaker labels; optionally accept a known speaker count when supplied.
- **Tests/checkpoint:** synthetic turns/overlap and boundary alignment unit tests; API persistence/reload of manual corrections; a labeled two-speaker local smoke test (including overlap/noise review) and explicit failure for missing model access. The UI now produces an auditable speaker-attributed transcript.

### Sprint 4 — Offline, evidence-linked structured minutes

- Add an internal Ollama adapter, initially evaluate a quantized Qwen3 4B Q4 model (Apache-2.0 base model; verify the exact downloaded quantization and license). Keep the model selectable by configuration. Bound context for a 30-minute transcript with chunked extraction/merge if necessary; request schema-constrained output with topics, summary, decisions, action items, owners, deadlines, and transcript-segment citations. Do not hallucinate unspecified fields; report invalid JSON, unavailable model, or unsupported context as failures rather than pretending completion.
- **Tests/checkpoint:** fake-LLM tests for missing owner/date, malformed/unsupported citations, chunk merging/deduplication, and model errors; local English/Italian smoke comparison against a human-checked reference. A recording can now produce an editable, source-linked draft of full minutes entirely offline.

### Sprint 5 — Review, export, and end-to-end usability

- Complete responsive plain-JS UI for transcript/audio navigation, minutes editing, visible uncertainty and source references, deliberate regeneration when transcript changes, save/reload, mark-reviewed action, delete, and clear progress/error states. A later React UI must only replace the client, not the API or processing logic.
- Export faithful TXT and versioned JSON including speaker-labeled transcript, minutes, evidence references, and explicit `DRAFT / UNREVIEWED` versus `REVIEWED` status. Permit draft downloads immediately; never imply an unreviewed export was approved.
- **Tests/checkpoint:** JSON-schema/round-trip and TXT-label tests; API edit/staleness/regeneration/review tests; browser end-to-end test from upload to correction, draft export, review, reviewed export, and deletion. This is the first fully usable MVP.

### Sprint 6 — Measured quality, model comparison, and release readiness

- Assemble a small, permissioned, non-sensitive evaluation set with human-verified transcripts, speakers, and decisions/actions across English/Italian, clean/noisy/overlapping audio. Measure WER, DER, precision/recall for decisions/actions/owner/date, citation validity, operator correction time versus manual minutes, total processing time, and peak VRAM/RAM. Establish the baseline first; only then set meaningful numeric acceptance thresholds and record them in the plan/tests. Guardrails (valid schema, no fabricated owner/date/citations, no third-party inference, no data past retention, no GPU OOM) apply from the outset.
- Benchmark faster-whisper INT8 against the proposed `whisper.cpp` Q3 turbo model **on the same recordings**, using a verified whisper.cpp-compatible Q3 weight; compare WER, speed, timestamp quality, and memory, not model file size alone. Implement/configure a second ASR adapter only if the benchmark supports keeping it; do not replace the baseline by assumption.
- Document clean-machine install, model download and offline run, GPU/CPU modes, recovery/deletion, known limitations (overlap/noise/hallucinations), license attribution, evaluation results, and how the Docker setup could later be migrated to an approved company VM with authentication/security work. Run unit/API/browser checks and the documented 30-minute local end-to-end hardware trial.
- **Checkpoint:** reproducible local Docker MVP and written evaluation against reference data; small verified release commits. No claim of production readiness or real-company-data authorization.

## Git and delivery workflow

1. After approval of this plan, add this document as root `plan.md`, create the initial documentation-only commit, and push it to the connected GitHub `main` branch. Do not start Sprint 0 until explicitly asked. The existing untracked `.idea/` directory belongs to the user; do not commit or modify it.
2. During implementation, make narrowly scoped commits with corresponding tests and a clear outcome; run the smallest relevant test set before each code commit, and keep CI green. Update root `plan.md` when checkpoints or measured quality gates change, in the related commit. Push commits to GitHub as agreed; never commit recordings, model weights, tokens, secrets, or local database state.

## Risks and decisions to validate during implementation

- The laptop's 6 GB VRAM is a constraint, not a guarantee: confirm CUDA/cuDNN/container GPU runtime compatibility, model fit, and throughput early; explicitly configured CPU mode remains available for development. If the proposed ASR/LLM pairing is too large, benchmark a smaller/swappable model rather than silently changing the requirements.
- Diarization requires accepting Hugging Face terms; cache model weights for offline inference and keep download credentials outside Git. Verify third-party model/package licenses before redistribution or VM migration.
- VAD may discard quiet speech, diarization can misattribute overlap, and LLMs can invent details. Preserve original audio/timestamps, surface uncertainty, require citations, and keep the operator in control.

References: the provided `Scheda_AI_Tirocinio_Riassunto_Riunioni.docx`; [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [Qwen3 4B GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF).
