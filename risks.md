# MeetScribe: architecture research and risk assessment

Research date: 2026-11-06. Assessed document: root `plan.md`.

## Executive assessment

The proposed local, single-user MVP is technically plausible and follows an established pattern: audio preprocessing, ASR, speaker diarization, timestamp reconciliation, and human correction, followed here by evidence-linked minutes generation. WhisperX and aTrain demonstrate much of the transcription portion of that pattern. They do not establish that the exact proposed models will fit, run fast enough, or produce reliable minutes on this laptop. [S01] [S02]

Keep the basic architecture: FastAPI, one sequential worker, SQLite, local files, and a private local model service. A distributed queue, vector database, agent framework, React rewrite, or second ASR implementation is not needed to answer the current requirements.

The most important unresolved questions are:

1. Whether the exact container/runtime/model combination works within **6 GB VRAM and approximately 15 GB host RAM**, including repeated transitions between ASR and Ollama.
2. Whether the full transcript reaches the minutes pipeline without context truncation, and whether extraction preserves corrections, negation, speaker ambiguity, and missing information.
3. Whether source citations and review status remain truthful after edits, regeneration, cancellation, deletion, and restart.
4. What the seven-day deletion promise actually means when the application or laptop is off, and which copies it covers.

This is a **design risk assessment, not an implementation audit or measured benchmark**. At assessment time the repository contains the plan but no application implementation. A risk below is not evidence that a defect or exploitable vulnerability already exists. Several risks already have sensible controls in the plan; this report identifies their residual risk and the concrete checks needed to make those controls effective.

## How comparable projects are built

The following is a synthesis of the cited implementations and documentation, not a statistical claim about every meeting assistant.

| Concern | Established approach and evidence | Fit for this MVP |
|---|---|---|
| Transcription pipeline | WhisperX combines VAD, faster-whisper, optional language-specific forced alignment, and pyannote speaker assignment. aTrain also combines local faster-whisper transcription and pyannote speaker detection. [S01] [S02] | The proposed cascade is reasonable. Reuse component capabilities rather than assembling duplicate preprocessing stages. |
| Timing and speakers | Word or segment timing must be reconciled with independently detected speaker turns. Community-1 provides both ordinary and exclusive diarization; exclusive output simplifies reconciliation but does not transcribe simultaneous voices. [S01] [S04] | Start with segment alignment and visible uncertainty. Add forced alignment only if measured timing or speaker-boundary errors justify another model. |
| Model execution | Quantized ASR reduces memory; model precision, batching, context, and model residency still affect actual requirements. Ollama can retain models and use CPU/GPU offloading. [S03] [S05] [S09] | Explicit compute settings and serialized model ownership are more important than advertised model file sizes. |
| Structured extraction | Ollama supports a JSON schema supplied to the request, followed by application-side validation. A valid JSON object is not proof of factual correctness. [S11] | Use the shared schema, verify references, retain uncertainty, and require human review rather than relying on prompting alone. |
| Long-meeting summarization | QMSum uses relevant transcript spans and a locate-then-summarize approach; MeetingBank provides segment-level transcript/minutes examples. [S29] [S30] | Evidence-preserving extraction before synthesis has useful precedent. These benchmarks do not prove that a quantized 4B model can reliably extract this application's owners, dates, and decisions. |
| Background processing | FastAPI distinguishes ordinary background work from heavy computation. Persistent job state is needed independently of an HTTP request's lifetime. [S16] | A database-backed job loop is sufficient locally. Keep expensive inference off the request/event-loop path; do not introduce a broker yet. |
| Persistence | SQLite is suitable for local state, but WAL still has one writer and cannot be shared across hosts through a network filesystem. [S18] | Keep transactions short and the database on a local persistent volume. Upgrade only when deployment or measured write contention requires it. |
| Quality assurance | Upstream projects document limitations in overlap, timing, language accuracy, and hallucination. Their results are not laptop-specific acceptance evidence. [S01] [S03] [S06] | Preserve original audio, provide efficient correction, and measure complete end-to-end usefulness, not just model throughput. |

### Reading the risk register

- **P0 - feasibility gate:** resolve early, before committing to the complete model stack.
- **P1 - MVP release gate:** implement or explicitly settle the policy before calling the local MVP complete.
- **P2 - usability/compatibility:** resolve before claiming support for the affected format or browser.
- **P3 - future deployment gate:** not a blocker for the authorized local test MVP; mandatory before expanding its boundaries.

Priorities express consequences and when a decision is needed, not measured probabilities. Each entry states a triggering use case, impact, mitigation, and a suggested verification. Source-backed component behavior is cited; failure scenarios and recommended controls are engineering conclusions for this plan.

## A. Runtime, models, and hardware

### R01 - Container GPU visibility does not prove library compatibility

**Priority / plan coverage:** P0. GPU preflight and pinning are planned; the working compatibility matrix is not yet demonstrated.

**Triggering use case:** Docker can see the RTX 4050, but ASR fails to load cuDNN, or installing pyannote upgrades PyTorch/TorchCodec to an incompatible combination.

**Issue and impact:** Docker Desktop GPU passthrough requires the Windows WSL 2 backend and suitable host drivers. faster-whisper/CTranslate2 has its own CUDA/cuDNN requirements; pyannote also needs a working audio-decoding stack. A successful `nvidia-smi` does not test any of those inference paths. Installation can consume a sprint before a single usable recording is processed. [S03] [S07] [S08] [S13]

**Mitigation:** Establish one tested matrix covering host driver, WSL/Docker backend, container image, Python, CTranslate2, CUDA/cuDNN, PyTorch, TorchCodec, FFmpeg, pyannote, and Ollama. Run Linux containers consistently. Do not mix independent "latest" installation examples or install a second driver stack inside WSL as an ad hoc fix.

**Verification:** In the final images, decode WAV and MP3, perform actual short GPU ASR and local diarization, and generate schema-constrained minutes. Record the exact versions and actionable failure messages.

### R02 - Model names and quantization labels are not interchangeable artifacts

**Priority / plan coverage:** P0. The plan correctly requires exact model/runtime verification; the artifacts still need to be identified.

**Triggering use case:** A downloaded "Q3 turbo" file is supplied to faster-whisper, an unrelated GGUF is tried with whisper.cpp, or a floating Ollama tag changes the Qwen quantization between evaluations.

**Issue and impact:** CTranslate2 models, whisper.cpp's documented model format, and Qwen GGUF weights are distinct artifacts. "INT8," "Q3," and "Q4" describe families of choices, not a portable runtime contract. Wrong assumptions cause load failures or invalid quality/memory comparisons. CTranslate2 also distinguishes mixed compute types such as `int8_float16`. [S03] [S05] [S23]

**Mitigation:** Record source repository and immutable revision, file hashes or model digest, conversion tool/version, tokenizer, quantization variant, runtime, and requested/effective compute type. Use an explicit supported GPU type, not an ambiguous promise that every operation is INT8. Verify the proposed whisper.cpp Q3 variant exists and loads in the selected build before benchmarking it.

**Verification:** Reject an incompatible artifact clearly; repeat a tiny inference from the recorded manifest after recreating containers. Keep model comparison settings and input recordings identical.

### R03 - Sequential stages do not guarantee sequential GPU residency

**Priority / plan coverage:** P0. ASR unloading is planned, but the reverse Ollama-to-ASR handoff also matters.

**Triggering use case:** Meeting A finishes summarization; meeting B immediately starts ASR while Ollama retains Qwen. Alternatively, the user requests minutes regeneration while another meeting is transcribing.

**Issue and impact:** Ollama documents retained model residency and independently configurable concurrency. Separate containers share the same physical GPU. Deleting a Python variable or clearing a PyTorch cache is not proof that CTranslate2 or another process released its allocations. The second job can OOM despite a successful first demo. [S03] [S09]

**Mitigation:** Serialize **all heavy operations**, including regeneration, through one scheduler. Explicitly unload Ollama before ASR, and release ASR before starting Ollama. Configure one Ollama request/model at a time and deliberate `keep_alive` behavior. Verify release with runtime/process telemetry; use a short-lived inference subprocess if the chosen library cannot reliably return memory.

**Verification:** Process several recordings back-to-back, interleave regeneration requests, and measure peak and post-stage VRAM. A single cold-start run is insufficient.

### R04 - Q4 model size is not the LLM's total VRAM requirement

**Priority / plan coverage:** P0. Context bounding and memory measurement are planned; a safe budget has not been established.

**Triggering use case:** A short transcript fits, but a long English/Italian transcript requires a larger context and Qwen partly offloads to CPU or fails.

**Issue and impact:** Model weights, KV cache, runtime buffers, and display/other-process allocations share 6 GB. Qwen's configuration has 36 layers, 8 KV heads, and head dimension 128. For illustration, a conventional unquantized FP16 KV cache alone is approximately `2 * 36 * 8 * 128 * 2 * context_tokens` bytes: **1.125 GiB at 8,192 tokens**, or **4.5 GiB at 32,768 tokens**, for one sequence, before weights and other memory. This is a calculation, not an Ollama measurement; actual allocation depends on backend/cache configuration. [S09] [S10] [S12]

**Mitigation:** Select explicit context/output limits from measured available memory. Prefer bounded extraction chunks over setting the model's maximum context. Inspect actual CPU/GPU placement; distinguish intentionally configured CPU operation from unexpected offloading. Evaluate KV-cache quantization only if needed and recheck extraction quality.

**Verification:** Measure the longest supported request, not just model loading. Record peak VRAM, host RAM, context, cache type, parallelism, and actual processor placement.

### R05 - CPU diarization and host memory can be the real bottleneck

**Priority / plan coverage:** P1. CPU diarization and bounded memory are planned; laptop resource budgets remain unmeasured.

**Triggering use case:** A 30-minute recording is decoded into multiple arrays while CPU diarization runs; Windows, Docker/WSL, an IDE, and a browser are already using host memory.

**Issue and impact:** Approximately 15 GB host RAM is not all available to containers. A 30-minute mono 16 kHz float32 waveform is about **115.2 MB**, but original stereo/high-rate arrays, intermediate tensors, model copies, and offloaded LLM layers multiply that footprint. CPU thread oversubscription, swap, laptop power limits, and sustained thermal throttling can make short-clip estimates misleading. CPU-default diarization is supported, not a throughput guarantee. [S04] [S14]

**Mitigation:** Decode once into a canonical representation where practical, avoid unnecessary copies, release stage buffers, and bound CPU threads/container memory with host headroom. Make CPU mode explicit and publish its own timings. Include model load/unload time and diarization in performance reporting.

**Verification:** Run the full 30-minute pipeline on the actual laptop and repeat it after warm-up under normal desktop load. Record per-stage elapsed time, memory, swap, and power mode.

### R06 - A cached model is not necessarily a complete offline installation

**Priority / plan coverage:** P1. Downloads, gated access, and persistent caches are planned; clean offline restoration needs proof.

**Triggering use case:** Setup downloads the principal weights, but the first offline job still requests a tokenizer, dependent checkpoint, configuration file, or decoder library. A new container looks in a different cache path.

**Issue and impact:** Community-1 requires accepting access conditions before setup. Hugging Face caches contain revision-specific files, and some loaders still make metadata requests unless explicitly offline. Partial downloads or different service-user directories can break the promised offline workflow. [S03] [S04] [S20] [S21]

**Mitigation:** Separate provisioning from inference. Download complete pinned artifacts and dependencies into explicit persistent locations, validate them, then use local paths/offline loader options. Keep setup credentials separate from model artifacts and remove them from runtime configuration when no longer needed. Provide an actionable "setup incomplete" readiness error rather than redownloading during a job.

**Verification:** Recreate containers from the provisioned volumes with external access blocked, then run English and Italian jobs. Repeat with one required artifact deliberately absent and confirm a clear failure.

### R07 - Local inference does not by itself mean zero external communication

**Priority / plan coverage:** P1. No remote inference is an explicit requirement; enforcement and telemetry policy are unspecified.

**Triggering use case:** A configurable adapter selects a cloud-backed diarization/model endpoint, or a library sends usage metadata during an otherwise local job.

**Issue and impact:** pyannote distinguishes locally executed Community-1 from remotely executed premium examples and documents optional telemetry including duration and speaker-count settings. Ollama also has cloud features; private inbound networking does not necessarily block outbound traffic. Audio upload to a remote backend would violate scope, while metadata telemetry is a separate policy question, not evidence that raw audio is sent. [S08] [S09] [S15]

**Mitigation:** Allowlist local adapters/models and service endpoints. Disable cloud features and unwanted telemetry in the pinned versions, including `OLLAMA_NO_CLOUD=1`, `PYANNOTE_METRICS_ENABLED=0`, and `HF_HUB_DISABLE_TELEMETRY=1` where applicable. Use Hugging Face offline mode and test runtime egress controls; do not assume "no published Ollama port" means offline.

**Verification:** Run a complete post-setup job with outbound traffic denied, inspect attempted connections without recording content, and ensure configured remote backends are rejected.

## B. Audio fidelity, timestamps, and speakers

### R08 - Upload caps do not bound parser, decoder, or temporary-file resources

**Priority / plan coverage:** P1. Type, size, duration, and corrupt-file validation are planned; enforcement needs to cover the full ingress path.

**Triggering use case:** A streamed upload omits or lies about `Content-Length`, a malformed WAV advertises a misleading duration, or many multipart parts fill temporary storage before endpoint validation runs.

**Issue and impact:** Extension/MIME checks and a 500 MB cap are insufficient alone. Reading the whole file into `bytes` consumes memory; `UploadFile` spools large uploads to disk, which is not an unlimited-resource solution. Decoder work and expanded audio can exceed what compressed size suggests. Availability can be lost even with accidental input, not just hostile files. [S17] [S24]

**Mitigation:** Enforce total request bytes, multipart count limits, and upload timeouts before or during parsing, not only after it. Count streamed bytes; cap decoded samples, duration, channels, and sample rate; bound probe/decode time and temporary storage. Use server-generated paths, maintained codecs, least-privilege execution, and no shell interpolation or remote input URLs.

**Verification:** Exercise missing lengths, excessive parts, truncated/oversized/corrupt media, misleading metadata, decoder timeouts, and cleanup after rejection. No job should be queued until validation succeeds.

### R09 - Normalization can discard channel information or desynchronize playback

**Priority / plan coverage:** P1. Resampling and original-audio preservation are planned; channel and playback policies are unspecified.

**Triggering use case:** A stereo recording has one participant per channel, phase-inverted channels cancel during averaging, or the inference decoder handles a WAV encoding that the browser cannot play.

**Issue and impact:** Community-1 automatically averages multichannel audio to mono. Averaging may lose useful source separation or attenuate speech. Independently generated playback and inference audio can also disagree in offset, especially with encoding delay or resampling. The operator may hear a different moment from the cited segment. [S04] [S25]

**Mitigation:** Inspect channel layout and duration before downmixing; document the supported normalization policy. Preserve the original and make irreversible transformations explicit. Warn/reject unsupported layouts instead of silently claiming full fidelity. If a browser-compatible playback derivative is necessary, generate it locally and verify its timebase against inference/original audio.

**Verification:** Include left-only/right-only speech, phase-inverted stereo, non-16-kHz audio, and supported WAV subtypes. Seek to events near both the start and end of an MP3.

### R10 - Conservative VAD can still remove the most important short utterance

**Priority / plan coverage:** P1. Conservative VAD is planned; configuration and interaction with ASR's own VAD need definition.

**Triggering use case:** A quiet participant says "no," "not approved," or a short deadline at a speech boundary; external Silero filtering and faster-whisper filtering are both applied.

**Issue and impact:** Lost speech cannot be recovered by downstream diarization or summarization. Double filtering and aggressive minimum-duration/silence settings can compound loss and transform a rejection into a decision. faster-whisper already integrates Silero VAD, so duplicate filtering is a concrete integration risk. [S03] [S22]

**Mitigation:** Choose one authoritative VAD path, retain padding around boundaries, and store the retained-region map. Tune for recall on quiet/short speech, not maximum silence removal. Distinguish genuinely empty speech from a processing failure, and allow explicit review/reprocessing with a less aggressive configuration.

**Verification:** Use human-labeled quiet speech, single-word turns, long silence, background noise, and boundary negations; check retention of those words, not only overall WER.

### R11 - Correct text can still have incorrect original-audio timestamps

**Priority / plan coverage:** P1. The plan correctly requires remapping; the invariant must survive every transformation.

**Triggering use case:** Multiple VAD regions are concatenated, ASR produces a segment across a join, and the UI seeks using the concatenated timeline while diarization uses the original timeline.

**Issue and impact:** Evidence links point to the wrong audio, speaker alignment shifts, and manual correction becomes unreliable. Whisper-style segment timestamps are not precise speaker-change boundaries; adding word alignment is possible but introduces language-specific models and still has unalignable-token limitations. [S01] [S03]

**Mitigation:** Keep an explicit piecewise mapping from retained samples to the canonical original timeline; split outputs crossing discontinuities. Use consistent units and preserve adequate precision until display. Align speaker turns only after remapping. Add forced alignment only when measured errors justify it, with English/Italian artifact coverage.

**Verification:** Use clips with known events after several removed silences, across region joins, and near 30 minutes. Assert monotonic, in-range timestamps and playback alignment using a documented tolerance.

### R12 - Exclusive diarization does not solve overlap or identify people

**Priority / plan coverage:** P1. Exclusive/regular output and uncertainty are planned; attribution remains probabilistic.

**Triggering use case:** Two people speak simultaneously, a short "I'll do it" overlaps another turn, or ASR places words from both speakers into one segment.

**Issue and impact:** Exclusive output chooses a non-overlapping primary attribution; it is not source separation and cannot recover words ASR missed. Speaker clusters can merge/split and labels can change on rerun. A wrongly attributed action can be assigned to the wrong person even when the words are correct. [S01] [S04]

**Mitigation:** Retain regular overlap evidence alongside exclusive alignment. Split segments at supported boundaries, allow unknown/ambiguous attribution, and avoid automatic ownership from uncertain speakers. Persist display-label mappings within each transcript revision; do not treat "Voice 1" as a cross-meeting identity or infer a name. Treat supplied speaker counts as fallible user input.

**Verification:** Include short turns, similar voices, noisy remote participants, overlap, and an intentionally wrong speaker-count hint. Evaluate attribution of action-bearing words, not only average DER.

### R13 - Language support does not guarantee accurate operational details

**Priority / plan coverage:** P1. English/Italian evaluation and hallucination warnings are planned; semantic edge cases need explicit coverage.

**Triggering use case:** An Italian meeting includes English product names, accented speech, a numeric target, a negated instruction, or long silence that produces plausible repeated text.

**Issue and impact:** Whisper documents hallucination, repetition, and uneven performance across languages and accents. A small WER change can hide a serious change to a name, amount, date, or "do not." Forcing a single language can also perform poorly on code-switching, and accidental translation changes what the transcript represents. [S06]

**Mitigation:** Make transcription versus translation explicit, preserve the declared language, document mixed-language limitations, and flag suspicious repetition/no-speech outputs. Treat model scores as review signals, not calibrated truth probabilities. Keep audio-based correction available and avoid silently "fixing" names through LLM rewriting.

**Verification:** Evaluate both languages separately, plus code-switching, names, numbers, dates, negations, silence, and poor audio. Check whether each error changes an extracted decision or action.

## C. Minutes generation and evidence

### R14 - Schema-valid, cited minutes can still be false

**Priority / plan coverage:** P1. Schema and citation validation are planned; those checks cannot enforce all factual guardrails.

**Triggering use case:** A model converts "we could ship Friday" into an approved decision and cites a real segment, or invents an owner while producing perfectly valid JSON.

**Issue and impact:** Schema validation proves shape; reference validation proves that an ID exists. Neither proves entailment, completeness, or correct interpretation. Even a citation to accurately transcribed words does not establish that a proposal became a decision. "No fabricated owner/date" cannot be guaranteed for arbitrary audio merely by making fields required or using constrained decoding. [S11]

**Mitigation:** Separate proposals, decisions, and actions in extraction rules; preserve explicit absence and uncertainty. Validate field types, allowed values, references, and supporting evidence. Require audio/transcript-grounded human review before treating minutes as authoritative; do not add another LLM and call it a proof. Measure unsupported claims as a release-quality metric.

**Verification:** Use proposed/rejected/conditional decisions and missing owners/dates. Require zero observed unsupported factual fields in the agreed acceptance set, while documenting that finite tests do not prove zero future hallucinations.

### R15 - The model's native context is not the runtime's usable context

**Priority / plan coverage:** P1. Chunking is anticipated; explicit token accounting and overflow behavior are not specified.

**Triggering use case:** A 30-minute transcript with timestamps, speaker labels, segment IDs, schema, and instructions exceeds Ollama's configured context, despite fitting an advertised Qwen context length.

**Issue and impact:** Ollama documents a 4k default for low-VRAM systems, while Qwen's model card advertises a much larger native window. Input, chat-template overhead, output allowance, and any thinking tokens all compete for a budget. A request may be truncated or fail depending on runtime behavior. Decisions near the start or end can disappear without a malformed JSON response. [S09] [S10] [S12]

**Mitigation:** Set context explicitly, budget with the actual tokenizer/template or a verified conservative bound, reserve output space, and chunk before overflow. Disable thinking through the supported runtime control when appropriate for extraction; validate the exact model's behavior rather than depending only on `/no_think`. Parse only final answer content and surface truncation/incomplete-generation errors.

**Verification:** Construct just-below/above-limit transcripts with uniquely identifiable actions at the beginning, middle, and end; prove every region is processed and no incomplete result is labeled complete.

### R16 - Chunk merging can erase corrections or fabricate consensus

**Priority / plan coverage:** P1. Extraction/merge and deduplication tests are planned; temporal semantics need explicit rules.

**Triggering use case:** An early chunk says "deliver Friday," a later one changes it to Monday, or an owner accepts a task several turns after the original proposal.

**Issue and impact:** Independent summaries lose cross-chunk references. Similarity-based deduplication can merge distinct tasks, keep both incompatible deadlines, or preserve a retracted decision. Recursive summarization can remove evidence at each stage. QMSum permits multiple nonadjacent relevant spans for one query, illustrating why contiguous chunks alone may miss needed context. These are engineering risks of bounded-context processing, not claims that a particular chunking library is defective. [S29]

**Mitigation:** Extract structured candidates from the original transcript with immutable evidence IDs and ordering. Use bounded overlap where helpful, reconcile explicit amendments/retractions, and preserve unresolved conflicts instead of guessing. Merge candidates rather than repeatedly summarizing summaries. Retain all supporting references when consolidating items.

**Verification:** Place acceptance, correction, negation, and owner assignment on opposite sides of chunk boundaries. Confirm the final item reflects the supported final state or clearly states the conflict.

### R17 - Anonymous speaker labels cannot reliably supply owners or calendar dates

**Priority / plan coverage:** P1. Unspecified fields are protected in the plan; identity and relative-date semantics remain unspecified.

**Triggering use case:** "Voice 2" says "I'll send it next Friday," someone mentions another person's name, or the recording is uploaded weeks after the meeting.

**Issue and impact:** Diarization does not establish a participant's real name. Upload time is not meeting time. A named person being discussed is not necessarily a speaker, and relative dates can depend on meeting date/timezone and local conventions. Incorrect normalization produces plausible but false accountability.

**Mitigation:** Keep anonymous speaker references distinct from explicitly stated owner names; do not invent their mapping. Preserve the source wording for deadlines. Only normalize a relative date when a reliable meeting date/timezone and an explicit normalization policy exist; otherwise retain ambiguity and the required "not specified" value where applicable. Do not use upload time as a silent substitute.

**Verification:** Include "I," "we," delegated tasks, mentioned nonparticipants, unknown meeting dates, locale-ambiguous dates, and uploads after the meeting. Check both JSON and displayed minutes.

### R18 - Recorded instructions and generated markup are untrusted content

**Priority / plan coverage:** P1. The LLM is designated untrusted; prompt-injection and browser rendering controls need definition.

**Triggering use case:** A participant discusses a prompt containing instructions, a transcript includes HTML-like text, or generated minutes contain an external image/link.

**Issue and impact:** Transcript text can influence model instructions, corrupting minutes despite valid JSON. Rendering text through unsafe HTML can execute script or load external resources, violating integrity or the offline boundary. With no tool access, prompt injection primarily threatens output; it does not automatically imply arbitrary host execution. [S26]

**Mitigation:** Treat transcripts as delimited data, isolate instructions from evidence, and test both supported languages. Give the LLM no tools, file access, credentials, or remote endpoints. Render content as text through safe DOM APIs; do not evaluate generated HTML/JavaScript. Keep the MVP plain-text and enforce a restrictive content-security policy rather than adding a rich-text surface.

**Verification:** Use harmless instruction-like dialogue and inert markup fixtures. Confirm they cannot invoke tools, change application state, run browser code, or trigger external requests; verify unsupported claims remain reviewable.

## D. State, processing, storage, and retention

### R19 - Edits can invalidate citations even when segment IDs still exist

**Priority / plan coverage:** P1. Stale minutes are planned; evidence versioning is not fully defined.

**Triggering use case:** An operator edits the text behind a cited ID, splits a segment, changes its speaker, or reruns ASR and reuses positional segment numbers.

**Issue and impact:** A citation can resolve successfully to content that no longer supports the generated statement. An ID based on list position is not durable provenance. Regeneration can also overwrite the evidence basis of manual edits while making the UI appear current.

**Mitigation:** Assign stable segment IDs within an explicit transcript revision. Tie each generated minutes revision to its source transcript revision and relevant pipeline configuration. Either keep the cited snapshot for that meeting's retention lifetime or make invalidated references visibly unresolved/stale. Use ordinary version columns/snapshots, not an elaborate event-sourcing system; purge every revision on deletion/expiry.

**Verification:** Edit text and speaker labels, split/merge segments, and rerun ASR. Confirm old citations never silently point to a new meaning and stale minutes cannot appear current.

### R20 - Review status must apply to the exact exported revision

**Priority / plan coverage:** P1. Draft labels and stale state are planned; all invalidating changes and concurrent saves need coverage.

**Triggering use case:** A reviewed document is edited, two browser tabs save conflicting changes, or generation started before an edit finishes afterward.

**Issue and impact:** A stale or changed document can retain `REVIEWED`, manual corrections can be lost, and TXT/JSON can disagree with the UI. A single `reviewed` boolean or one status enum is insufficient to express processing state, source staleness, content revision, and review validity.

**Mitigation:** Tie review to the saved transcript/minutes revision pair and invalidate it after relevant edits/regeneration. Keep job state, staleness, and review state distinct. Use optimistic revision checks for saves and generation completion; stage a new generated draft rather than replacing manual work. Derive export labels from authoritative current state, including stale results.

**Verification:** Test reviewed-then-edit, edit-during-generation, two-tab conflicts, export during save, and failed regeneration. Anything not reviewed in its current form must export conspicuously as `DRAFT / UNREVIEWED`.

### R21 - One sequential worker can still block the API or become several workers

**Priority / plan coverage:** P1. Sequential scheduling is planned; process topology needs to enforce it.

**Triggering use case:** Synchronous inference runs inside an async request handler, or multiple Uvicorn processes each start their own queue loop.

**Issue and impact:** Upload/status/playback/cancellation can stop responding during inference. Conversely, adding API workers can accidentally multiply GPU jobs. In-memory background tasks also do not themselves implement durable scheduling or recovery. [S16]

**Mitigation:** Return a persisted job ID, run a single explicitly owned processing loop outside the request path, and keep inference from blocking the ASGI event loop. For the MVP, one API process with a dedicated worker execution context and database job state is enough. Configure production/reload behavior deliberately; use atomic claims and a singleton mechanism before adding processes.

**Verification:** During actual inference, poll status, seek audio, edit a different meeting, and request cancellation. Attempt a second application instance and confirm no duplicate job execution.

### R22 - Restart, retry, cancellation, and deletion can resurrect results

**Priority / plan coverage:** P1. Recovery and deletion safeguards are planned; race semantics must be explicit.

**Triggering use case:** The laptop sleeps or Docker stops after inference but before commit; a user deletes a running meeting while its model call is still completing.

**Issue and impact:** Jobs may remain processing forever, execute twice, lose manual edits, or recreate files and rows after deletion. Cancelling an HTTP request does not necessarily stop native inference or release GPU memory. SQLite cannot atomically commit a filesystem operation with a row update.

**Mitigation:** Use explicit attempt/revision IDs, atomic job claims, bounded retries for transient errors, and stage-boundary checkpoints. Mark cancellation/deletion before cleanup and reject late results whose attempt or meeting is no longer valid. Publish stage artifacts via temporary names and atomic rename, reconcile orphan files at startup, and treat cleanup failure as visible/retryable. Do not restart a still-active attempt without establishing its owner is gone.

**Verification:** Inject interruption before/after artifact publication and DB commit; delete during each stage and during generation. Confirm no resurrection, duplicate active job, partial "complete" result, or overwrite of reviewed work.

### R23 - Long SQLite transactions and naive backups undermine reliability

**Priority / plan coverage:** P1. SQLite and migrations are planned; transaction, durability, and backup policies are unspecified.

**Triggering use case:** A transaction stays open throughout diarization, progress updates collide with edits/cleanup, or the database file is copied while WAL contains committed changes.

**Issue and impact:** WAL allows reader/writer concurrency but still only one writer; long transactions cause lock contention and WAL growth. Copying the main database alone is not a generally safe live backup. A storage-backed migration does not make an older application automatically compatible with the new schema. [S18] [S19]

**Mitigation:** Keep transactions short and never hold one around inference. Use deliberate WAL/busy-timeout, per-connection foreign-key enforcement, checkpoint, and durability settings. Keep the DB on local storage, not a network share. Version migrations and fail clearly on incompatible schema. If backups are introduced, use SQLite's backup API or a quiesced consistent snapshot and apply the same retention policy.

**Verification:** Exercise concurrent save/progress/cleanup, disk-full and restart behavior, migration from the previous schema, and restoration from a supported backup. Do not add archival backups silently to a seven-day-retention app.

### R24 - Per-file limits do not bound queue length or total disk use

**Priority / plan coverage:** P1. One-at-a-time execution and retention are planned; admission limits are not.

**Triggering use case:** An operator uploads many individually valid 500 MB recordings while one slow job is processing; model caches and temporary audio already occupy much of the Docker disk.

**Issue and impact:** Seven-day retention limits age, not upload rate. Twenty maximum-size uploads alone consume about **10 GB decimal**, excluding derivatives, models, database, and build layers. A full host/WSL disk can break uploads, commits, cleanup, and unrelated applications. Queue wait also grows even though only one job runs.

**Mitigation:** Set an explicit outstanding-job limit, aggregate storage quota, reserved free-space threshold, and bounded temporary/model-cache usage. Reject excess work before consuming the upload where possible and explain retry conditions. Include regeneration in admission control; avoid creating duplicate work from repeated clicks.

**Verification:** Fill the queue/quota and simulate low disk space; check deterministic rejection, partial-file cleanup, and continued availability of review/delete. Report queue position without promising an unmeasured completion time.

### R25 - A stopped application cannot guarantee deletion at exactly seven days

**Priority / plan coverage:** P1 policy decision. Startup and interval cleanup are planned; the absolute retention wording is stronger than those mechanisms.

**Triggering use case:** The laptop is off on day seven and resumes on day ten, or a cleanup interval expires while the application is stopped.

**Issue and impact:** Startup cleanup cannot erase data while the system is not running. Periodic cleanup also adds interval lag while running. Clock errors and active jobs crossing expiry can further violate "no data past retention." This is a guarantee mismatch, not a problem solved by UTC alone.

**Mitigation:** Explicitly choose/document the guarantee. The simplest local policy is immediate access denial at `uploaded_at + 7 days`, cleanup within a stated running-time interval, and cleanup before serving content or resuming jobs after startup. If physical removal at the deadline is mandatory, an independently running cleanup mechanism and available host are required; a powered-off host still needs a different guarantee. Do not silently relax the plan.

**Verification:** Use a controlled clock, expiry during processing, missed intervals, and restart after expiry. Ensure neither playback, detail, export, nor worker recovery exposes expired content before cleanup finishes.

### R26 - Deleting records and files does not erase every copy

**Priority / plan coverage:** P1 for scope and logical deletion; stronger physical-erasure assurance needs an explicit policy.

**Triggering use case:** A meeting is deleted but text survives in SQLite free pages/WAL, an orphan temp file, browser storage, logs, a snapshot, or an exported TXT in Downloads.

**Issue and impact:** SQLite normally reuses deleted space rather than immediately overwriting it. Database cleanup does not erase SSD remapping, host backups, Docker virtual-disk history, or user exports. Model/setup credentials can also persist if stored in a shared model cache. "Anonymous speakers" and "deleted in the app" are not full data-erasure guarantees. [S19] [S20] [S21]

**Mitigation:** Inventory application-managed copies, including transcript revisions, playback derivatives, failed jobs, caches, and content-bearing diagnostics. Keep browser persistence content-free and use appropriate no-store responses. Delete all managed artifacts with retryable, visible failures; evaluate SQLite secure deletion/checkpoint/VACUUM behavior for the selected policy. Use host disk protection and controlled backups, but do not claim these prove forensic erasure. Explain that downloaded exports leave app control.

**Verification:** Seed identifiable synthetic markers, delete/expire the meeting, and inspect managed storage and logs, including interrupted-job paths. Check backup/export policy separately; never promise recall of already downloaded files.

## E. Compatibility, scope, and delivery

### R27 - Playback and export require contracts beyond "WAV/MP3" and "JSON"

**Priority / plan coverage:** P2. Playback and versioned exports are planned; supported encodings, seeking, and field semantics need definition.

**Triggering use case:** A valid WAV cannot play in a target browser, a 30-minute MP3 must download fully before seeking, or an Italian export loses accents or changes unspecified values.

**Issue and impact:** Decoder acceptance and browser codec support differ. A generic stream need not implement byte ranges; Starlette's file response does support range requests. Vague time units, nullable/sentinel values, revision IDs, and review metadata make JSON consumers incompatible even when `/api/v1` stays unchanged. [S25]

**Mitigation:** Define a small supported browser/audio profile and use the framework's range-capable file response through a meeting-ID lookup, never arbitrary paths. Specify UTF-8, timestamp units/precision, schema version, missing-value representation, source revisions, and review/stale semantics. TXT and JSON should render the same saved state. Serve assets locally without CDN dependencies.

**Verification:** Test seeking with `206`/invalid ranges, playback of supported WAV/MP3 variants, Unicode round trips, old-version fixtures, draft/stale labels, and fully offline UI loading.

### R28 - Loopback binding is necessary but not a complete trust boundary

**Priority / plan coverage:** P1. Loopback-only publishing and private Ollama are planned; browser-origin and container-binding rules are unspecified.

**Triggering use case:** Compose publishes the app on all interfaces, the app binds only to container loopback and becomes unreachable, or an unrelated website attempts requests to the local API.

**Issue and impact:** Host port publishing and in-container listening addresses are different. Localhost is not user authentication, and browser same-origin rules are not a universal prohibition on sending state-changing requests. Docker also documents a localhost-publishing caveat in releases older than 28.0.0. Exposure may permit unwanted jobs, state changes, or access from other local processes/users. [S15] [S27]

**Mitigation:** Bind the server to the required container interface while publishing only to host `127.0.0.1`; do not publish Ollama. Address it by Compose service name, not app-container `localhost`. Validate allowed Host/Origin values, reject cross-site mutations with appropriate CSRF protections, avoid state-changing GETs and permissive CORS, and use a maintained Docker version. Document the trusted-local-machine assumption.

**Verification:** Confirm host UI access, app-to-Ollama access, no intended LAN/IPv6 exposure, and rejection of hostile-origin requests on supported browsers. CORS configuration alone is not the check.

### R29 - Anonymous labels are not anonymization or authorization

**Priority / plan coverage:** P1 for MVP scope enforcement; legal/security review is a future real-data gate.

**Triggering use case:** A user uploads an ordinary workplace recording because the UI labels participants "Voice 1," or exports names and sensitive discussion present in the transcript.

**Issue and impact:** Voices, spoken names, and contextual details can identify people. Local processing does not by itself remove personal-data obligations or authorize recording. The plan's restriction to synthetic or explicitly authorized non-sensitive tests is essential, not merely a deployment preference. European Commission guidance treats storage, transcription-related processing, disclosure, and erasure as data processing. [S28]

**Mitigation:** State the authorized-test-only boundary in setup, upload UI, and documentation; do not describe the output as fully anonymized. Do not infer identity or sensitive traits. Before real company data, obtain organization-specific approval covering recording authorization/lawful basis, participant information, access, retention, exports, and incident handling. This report is not a legal determination.

**Verification:** Review every user-facing privacy claim and export field against the actual data retained; keep fixtures and evaluation recordings permissioned and non-sensitive.

### R30 - License pinning and dependency pinning require different maintenance work

**Priority / plan coverage:** P1. Attribution, gating, and pinned dependencies are planned; redistribution and update processes remain to be established.

**Triggering use case:** A Docker image containing weights is shared, a third-party quantization is adopted, or a pinned decoder with a subsequently fixed security issue remains indefinitely deployed.

**Issue and impact:** The app's license does not govern every package, model, conversion, codec build, or container base. Gated access conditions are not the same as a redistribution license. Conversely, reproducibility through pinning does not make old dependencies safe or supported. Blind updates can break the working GPU matrix. [S04] [S08] [S12] [S13] [S24]

**Mitigation:** Maintain an inventory of exact artifacts, provenance, applicable licenses/notices and gated conditions; verify them before redistribution or VM migration. Prefer trusted releases, integrity verification, and separate provisioned weights over embedding everything in an application image. Keep credentials out of build layers/logs. Schedule controlled security updates with compatibility and evaluation checks, not unattended "latest" upgrades.

**Verification:** Reproduce the image/model manifest, inspect distribution contents and notices, and rerun targeted inference/decoder checks after an update. Do not infer a quantized artifact's terms solely from the base model name.

### R31 - Fake-model CI and average scores can hide a non-useful product

**Priority / plan coverage:** P1. The planned evaluation is strong, but most measurements are deferred to Sprint 6.

**Triggering use case:** Unit/API tests pass with fakes and a two-speaker clean demo succeeds, but noisy Italian meetings require more correction time than writing minutes manually.

**Issue and impact:** Fakes test contracts, not actual GPU fit, model output shape, download completeness, or factual quality. Aggregate WER/DER can conceal errors in action-bearing words and cannot establish decision/action usefulness. Upstream throughput measurements use other hardware/settings; comparing different beam sizes, batching, quantization, or output quality produces misleading winners. [S01] [S03] [S06]

**Mitigation:** Keep lightweight CI, but introduce a tiny human-checked real-model contract/evaluation set early, then expand it. Stratify by language, noise, overlap, duration, and speaker count. Report raw error counts and critical negation/name/date/owner errors, citation support as well as ID validity, correction time, stage timings, and full resource peaks. Compare against manual minutes on equivalent material; define WER normalization and DER overlap/collar settings. QMSum/MeetingBank can inform test structure, but do not replace representative English/Italian audio or establish reuse permission for every recording. [S29] [S30]

**Verification:** Freeze a held-out acceptance set and thresholds after baseline measurement; record zero-observed invariant violations without claiming universal correctness. Compare ASR alternatives on identical audio/settings, including load time and resulting minutes quality, not weight size or ASR speed alone.

### R32 - A company VM is a new deployment boundary, not just a Compose move

**Priority / plan coverage:** P3. Correctly out of scope today; migration path must avoid implying production readiness.

**Triggering use case:** The app moves to a CPU-only VM, several users submit jobs, API replicas each start a worker, or SQLite is placed on shared storage.

**Issue and impact:** GPU passthrough may be absent, explicit CPU mode may be too slow, the sequential queue becomes a service bottleneck, and local-trust assumptions cease to hold. WAL cannot be shared across hosts through a network filesystem. Multiuser access also needs per-meeting authorization, isolation, transport security, and operational ownership. [S07] [S16] [S18]

**Mitigation:** Preserve framework-independent schemas, model adapters with real uses, stable IDs, and migrations, but do not implement distributed infrastructure now. Before migration, benchmark approved hardware and define capacity/service objectives; add authentication/authorization, TLS, tenant/user isolation, approved storage, backup/retention operations, and monitoring. Move to a broker/PostgreSQL/object storage only when independent workers, multiple hosts, or measured contention require them.

**Verification:** Require a separate deployment review and load test before accepting company data or remote users. A laptop success and a container health check are not that approval.

## Capacity and scaling implications

The intended single-job limit is a useful safeguard, not an accidental deficiency. For a sustained workload, let:

- `D` be average recording duration in minutes;
- `r` be measured processing minutes per audio minute for the complete pipeline;
- `L` be per-job setup/load overhead in minutes.

Then mean service time is approximately `T = D * r + L`, and one continuously available worker has a theoretical maximum of `60 / T` jobs/hour. This excludes downtime and variation; acceptable interactive latency requires spare capacity. For example, **if** a 30-minute meeting takes 30 minutes to process, capacity is at most two jobs/hour, and the fourth of four simultaneous submissions waits roughly 90 minutes before starting. This is an illustration, not a laptop prediction.

| Scaling pressure | First sensible response | When a larger change becomes justified |
|---|---|---|
| Occasional local upload burst | Queue cap, clear status/cancel, storage quota; keep one GPU job. | Demand persistently exceeds measured capacity and waiting is unacceptable. |
| Longer recordings | Keep the 30-minute cap; first measure chunking, memory, and review effort. | A separately approved longer-duration feature with its own acceptance tests. |
| Slow CPU diarization | Measure thread/resource settings and stage share; consider serialized GPU diarization only after a fit/quality trial. | Diarization demonstrably dominates and cannot meet the required turnaround on approved hardware. |
| Longer LLM input | Bounded evidence-preserving extraction/merge and explicit token budgets. | Evaluation proves that more context is necessary and approved memory can support it. |
| More UI/API traffic | Keep request handlers responsive, reduce polling, paginate lists. | Measured API demand requires independent API processes; worker ownership must be solved first. |
| More simultaneous writers/hosts | Short local transactions and a single DB-backed worker. | Independent services or sustained write contention justify a server database and durable broker. |
| New operating system/GPU | State the supported platform explicitly; offer deliberate tested CPU mode. | Add a backend only after verifying its runtime, models, quality, and packaging; Docker does not make NVIDIA acceleration universal. |

## Recommended additions to the implementation checkpoints

These are recommendations for the plan, not changes to `plan.md` or authorization to begin implementation.

| Timing | Required evidence or decision | Risks addressed |
|---|---|---|
| Early Sprint 0 feasibility work | Exact compatibility/artifact manifest; short real-model execution; repeated ASR-to-Ollama-to-ASR handoffs; explicit context/placement limits. | R01-R05 |
| Before claiming offline operation | Recreated-container execution with external access blocked, complete caches, cloud/telemetry policy, and no runtime credentials unnecessarily retained. | R06-R07, R30 |
| Sprint 1 lifecycle work | Stream/parser/decode resource limits; persistent job ownership; deletion/cancel/restart fault injection; disk/queue admission; agreed retention wording. | R08, R21-R26 |
| Sprints 2-3 quality work | Original-timebase fixtures, quiet negations, stereo/channel cases, English/Italian/code-switching, overlap, unknown attribution, and target-browser playback. | R09-R13, R27 |
| Sprint 4 extraction work | Token-overflow detection, cross-chunk amendments, evidence support checks, identity/date ambiguity, actual runtime thinking/output behavior, and untrusted-content fixtures. | R14-R18 |
| Sprint 5 review/export work | Revision-bound citations/review; two-tab and edit-during-generation cases; faithful draft/stale/reviewed TXT/JSON; host/origin restrictions. | R19-R20, R27-R29 |
| Sprint 6 release evidence | Held-out quality/correction-time results; repeated full 30-minute hardware runs; quota, offline, retention, and supported-browser checks. | All P0/P1 risks, plus claimed P2 support |
| Only before remote/company deployment | Approved infrastructure, real-data authorization, user access model, security/operations review, and measured load capacity. | R29-R32 |

Do not invent laptop throughput, acceptable WER/DER, or correction-time thresholds in advance. Establish representative baselines and then agree the thresholds, as the plan proposes. In contrast, contract invariants can be checked immediately: invalid citations are rejected; unsupported factual fields fail the acceptance fixtures; stale/current review labels are truthful; managed expired content is inaccessible; external inference is disabled; and a job exceeding a resource budget fails visibly rather than pretending completion.

## Sources and evidence limits

All sources below were consulted on 2026-11-06. They are primary project documentation, author-published benchmark descriptions, model cards, or official guidance. Many are mutable; an implementation must pin its own versions instead of treating these links as a frozen compatibility specification. Source descriptions identify the relevant material, not blanket endorsement of every upstream performance or marketing claim.

| ID | Source | Relevant evidence |
|---|---|---|
| S01 | [WhisperX](https://github.com/m-bain/whisperX) | Cascaded ASR/alignment/diarization architecture; language-specific alignment; overlap and timing limitations. |
| S02 | [aTrain](https://github.com/aTrainTranscription/aTrain) | Comparable local transcription application using faster-whisper and pyannote; playback-oriented export and platform considerations. |
| S03 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CUDA/cuDNN requirements; PyAV decoding; integrated Silero VAD; compute settings; benchmark hardware and fair comparisons. |
| S04 | [pyannote Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1) | Mono 16 kHz processing/downmixing; CPU default; gated setup; offline operation; exclusive and regular diarization. |
| S05 | [CTranslate2 quantization](https://opennmt.net/CTranslate2/quantization.html) | Supported compute/quantization types and mixed floating-point layers. |
| S06 | [Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo) | Pruned turbo model; long-form processing; language/accuracy variation; hallucination and repetition caveats. |
| S07 | [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/) | NVIDIA GPU passthrough prerequisites on Windows/WSL 2. |
| S08 | [pyannote.audio](https://github.com/pyannote/pyannote-audio) | TorchCodec/FFmpeg setup; local versus remote pipelines; telemetry fields and configuration. |
| S09 | [Ollama FAQ](https://docs.ollama.com/faq) | Model residency/unloading; concurrency and queue settings; CPU/GPU placement; cache quantization; cloud-disable control. |
| S10 | [Ollama context length](https://docs.ollama.com/context-length) | Runtime context defaults based on available VRAM and memory/offload implications. |
| S11 | [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs) | JSON-schema request format and application-side validation. |
| S12 | [Qwen3-4B GGUF model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [model configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json), and [Ollama thinking controls](https://docs.ollama.com/capabilities/thinking) | Architecture/quantization/context and thinking behavior; configuration inputs for the illustrative KV-cache calculation. |
| S13 | [TorchCodec](https://github.com/meta-pytorch/torchcodec) | PyTorch/Python compatibility table, FFmpeg shared-library requirements, and platform-specific wheels. |
| S14 | [Docker Desktop WSL 2 backend](https://docs.docker.com/desktop/features/wsl/) | Windows/Linux resource and environment boundaries. |
| S15 | [Docker port publishing](https://docs.docker.com/engine/network/port-publishing/) and [Compose networking](https://docs.docker.com/compose/how-tos/networking/) | Host/container binding; localhost caveats; service discovery; internal networks versus external connectivity. |
| S16 | [FastAPI background tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) | Request-adjacent tasks versus heavy background computation; limits of framework task scheduling. |
| S17 | [FastAPI request files](https://fastapi.tiangolo.com/tutorial/request-files/) | Full-memory `bytes` uploads versus disk-spooled `UploadFile`. |
| S18 | [SQLite WAL](https://sqlite.org/wal.html) | Single writer, checkpoints, durability choices, WAL sidecars, and same-host filesystem requirements. |
| S19 | [SQLite VACUUM](https://sqlite.org/lang_vacuum.html) and [online backup API](https://sqlite.org/backup.html) | Deleted-page remnants, database compaction limits/costs, and consistent database backups. |
| S20 | [Hugging Face cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) | Revision-specific artifacts, cache locations, and snapshot completeness. |
| S21 | [Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) | Offline behavior, telemetry, token storage, and Windows cache considerations. |
| S22 | [Silero VAD](https://github.com/snakers4/silero-vad) | Supported input/runtime paths and audio-loading dependencies. |
| S23 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Documented Whisper model format, quantization, model-specific loading, and CLI audio requirements. |
| S24 | [OWASP file upload guidance](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html) | Layered validation, resource limits, generated paths, parser exposure, and least-privilege storage. |
| S25 | [Starlette responses](https://starlette.dev/responses/) | Framework file streaming and byte-range behavior. |
| S26 | [OWASP LLM prompt injection guidance](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) | Untrusted document instructions, limited prompt-only defenses, least privilege, and generated HTML/Markdown risks. |
| S27 | [OWASP CSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html) | Cross-origin state changes, request validation, tokens, and browser-origin limitations. |
| S28 | [European Commission: data protection explained](https://commission.europa.eu/law/law-topic/data-protection/data-protection-explained_en) | Personal-data processing includes storage, adaptation, disclosure, and erasure; local technology does not determine legal applicability. |
| S29 | [QMSum author repository](https://github.com/Yale-LILY/QMSum) | Query-focused meeting summaries, potentially nonadjacent relevant spans, locate-then-summarize implementation, and separate evaluation splits. |
| S30 | [MeetingBank project](https://meetingbank.github.io/) | Meeting transcripts aligned with minutes, agenda/metadata, and segment-level summarization examples; a different domain/duration distribution from this MVP. |

Unverified by this research: actual laptop performance/memory peaks, the final dependency lock, exact selected quantization digests and their redistribution terms, the contents of the referenced DOCX (not present in the repository), and the authorization/policy requirements of any future company deployment.
