# Sprint 3.4 — Bounded minutes extraction

## Outcome

Added `MinutesExtractor` as a replaceable adapter boundary and
`OllamaMinutesExtractor` for local Qwen3 extraction. The extractor sends the
complete aligned transcript in one request, including segment IDs, millisecond
timestamps, anonymous speaker IDs and text. It returns a typed draft only
after the final response passes the 3.3 schema and evidence checks.

The adapter is not yet connected to the job API or browser. There is no
real-model minutes-extraction result in this slice.

## Context and completion bounds

The extraction context remains at **4,096 tokens**, matching the measured
smoke/handoff configuration in slices 3.1 and 3.2. It does not use Qwen3's
advertised maximum or make a claim about the unknown future VM. The manifest
reserves at most **768 generated tokens** and **128 tokens** for chat-template
overhead; temperature is zero. The existing Ollama client disables thinking
and sends the manifest's explicit `num_ctx`.

The preflight prompt bound is the UTF-8 byte length of the complete prompt,
including the instructions, compact output-shape example and JSON-serialized
transcript, plus the 128-token template reserve. The complete input bound and
768-token generation reserve must fit within 4,096 tokens. The full Pydantic
JSON schema is sent separately as Ollama's structured-output format; the
prompt's output-shape example is included in the byte bound. This byte-based
bound is deliberately conservative, not a model tokenizer measurement. On
every extraction, the adapter also requires Ollama's prompt and generation
counts and rejects missing counts, a prompt count above the bound, generation
above 768, or a combined count above 4,096. A real extraction is still needed
to verify that the 128-token template reserve is sufficient in practice.

IMPORTANT:
An input that does not fit fails before model inference. This slice does not
chunk or silently drop transcript segments, so it cannot produce a
success-shaped partial draft. The result's `included_segment_ids` records
which segments were sent; it does not prove that the model understood every
segment.

## Prompt and validation boundaries

Transcript text is serialized as JSON and explicitly treated as untrusted
recorded content, not instructions. The prompt directs the model to preserve
negations, changed proposals and deadlines, distinguish unknown attribution
from names, and avoid unsupported facts. The model may still misinterpret
speech or ASR errors; valid citations establish traceability, not truth.

Only a response ending with `done_reason == "stop"` is considered a normal
generation completion. In the pinned Ollama version, `"stop"` is the
successful completion reason used by the local generation path, while
`"length"` indicates that a generation limit was reached. The code names
`"stop"` as `OLLAMA_DONE_REASON_STOP` in `app/adapters/ollama_client.py`. Ollama's
public API describes `done_reason` as a string, not an enumerated contract;
its meaning should be rechecked when upgrading Ollama.

This completion reason is specific to the Ollama adapter, not a universal LLM
provider value. If the provider or API changes, the replacement adapter must
interpret its own completion status; the success value may need to change or
be made provider-specific rather than assumed to be `"stop"`. Even `"stop"`
does not guarantee a valid or accurate draft: the adapter still parses and
validates the final response, checks runtime accounting, and propagates
failures rather than returning a partial result. JSON shape, completeness,
citations and deadline/owner evidence use the independent checks introduced
in slice 3.3.

## Validation

- `test/unit/test_minutes_extractor.py`: **18 passed** using a fake model
  client.
- Default suite: **251 passed, 30 skipped**, 13.92 seconds; one existing
  Starlette/AnyIO deprecation warning.
- Ruff, formatting, mypy, `uv lock --check` and `git diff --check`: passed.
- No real Ollama extraction, browser test, or long GPU/audio-to-minutes
  integration test was run. The real runtime token-count comparison and
  template reserve remain unverified with an actual minutes generation.
- Existing ASR exact-text failure and repeated-sentence XFAIL are unchanged;
  this slice did not rerun the opt-in ASR accuracy tests.

## Files and next step

The implementation is in `app/core/minutes_extractor.py` and
`app/model_manifests/summary_model.toml`, with fake-client coverage in
`test/unit/test_minutes_extractor.py`. `README.md`, `plan.md` and
`docs/status/test-status.md` record the current status. Slice 3.5 connects minutes
generation to the application workflow and implements summary-only editing
and export; the browser transcript display is scheduled for removal there.
