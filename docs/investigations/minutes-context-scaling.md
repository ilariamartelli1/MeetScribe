# Minutes context scaling and limitations

This note records current context-budget and scaling evidence for the minutes
workflow. Expand it when resuming slice 3.7's context investigation. It is not
a capacity guarantee or approval to change the current fail-fast behavior.

## Current extraction path

ASR and diarization produce attributed transcript segments. The extractor
serializes **every segment** as JSON with its ID, timestamps, speaker ID and
text, then sends the complete transcript in one request. It does not truncate
or chunk the transcript. The output schema constrains the response fields; it
does not hide input segments. JSON formatting adds some prompt overhead.

The current context budget is **16,384 tokens**, with a maximum of **4,096
generated tokens** and a **128-token** chat-template reserve. The preflight
prompt bound is its UTF-8 byte length plus the template reserve; that bound and
the generation reserve must fit within 16,384. This is a conservative estimate,
not tokenizer output. The full Pydantic response schema is sent separately as
Ollama's structured-output format. The latest English MP3 run reported a
10,199-token prompt upper bound plus 4,096 reserved output tokens, for a
conservative total of 14,295.

If the bound exceeds the context size, extraction fails before model
inference with `MinutesContextBudgetError`. No partial draft or silent
segment-dropping fallback is produced.

## Why the budget exists and how it relates to the GPU

The context window covers the prompt and generated response, not just
transcript text. Instructions, the output-shape example and serialized segment
metadata use context too. `num_predict=1024` reserves bounded room for a
complete minutes response; if the model stops early or reaches its generation
limit, the extractor rejects the result rather than exporting partial JSON.
The full transcript must fit, so the app does not silently drop early or late
segments to make a request pass.

The limit also bounds inference cost. With other settings fixed, a larger
context generally needs a larger KV cache, increasing GPU or host memory use
and potentially latency. The exact cost depends on the model, quantization and
runtime. **16,384 tokens is not a measured maximum imposed by the RTX 4050's
6 GB VRAM**: no context-capacity sweep was performed. The latest English MP3
run sampled a 4,710 MiB GPU peak out of 6,141 MiB, but a single sampled run
does not prove a safe hardware boundary or a necessary minimum. CPU and future
VM deployments also need their own resource measurements.

In the first 3.7 run at the former 4,096-token setting, the **11,322-token**
figure was a preflight upper bound, and inference never started. It does
**not** prove that the GPU ran out of memory or that the tokenizer would count
the prompt as 11,322 tokens. The byte-based bound may exceed the runtime
tokenizer count.

## Observed limit

In the first approved slice 3.7 run, a **297.305-second English recording**
transcribed successfully, but its conservative total context estimate was
**11,322 tokens**, above the **4,096-token** budget. The model was not called;
this is evidence of a context-budget rejection, not an LLM quality failure.
The Italian and two-voice diagnostic recordings were not reached. No actual
generation token counts, draft-quality result, correction-time measurement
or minutes-stage resource report came from that run.

On **2026-11-23**, the English WAV checkpoint completed with a 12,288-token
context and 1,024-token output limit. A first retry with the old 768-token
output limit stopped at exactly 768 generated tokens (`reason=length`), not
at the context preflight. The final WAV run used **2,901 prompt tokens** and
**812 generated tokens** against a **10,290-token prompt upper bound**;
transcription took **53.767 s** and minutes generation **67.259 s**. It passed
in **126.09 s** overall, with a sampled GPU peak of **4,444 MiB** during
minutes generation and **2,294 MiB** during transcription, out of **6,141
MiB** total.

The later English MP3 run completed with a 16,384-token context and a
4,096-token output limit. The **297.305-second** recording used **2,862 prompt
tokens** and **1,023 generated tokens** against a **10,199-token prompt upper
bound**; transcription took **72.659 s**, minutes generation **120.283 s**,
and the test passed in **203.09 s** overall. Sampled GPU use peaked at
**4,710 MiB** during minutes generation and **2,294 MiB** during
transcription, out of **6,141 MiB** total. This is one sampled run; brief
resource spikes may be missed. The MP3 test verifies a complete response and
export shape, not semantic minutes quality. Italian and known-ASR-error cases
were not run with the current settings.

This does not establish a fixed maximum meeting duration. Transcript token
count varies with language, speaking rate, vocabulary and transcript content;
segment metadata and prompt instructions also consume budget. A successful
upload or transcription therefore does not guarantee that minutes extraction
fits. The current five-minute/100 MB upload limit is separate from the
summary-context limit. The English fixture failed at the former 4,096-token
setting, the WAV passed at 12,288 tokens and the MP3 passed at 16,384 tokens;
none of these results establishes a universal five-minute capacity.

## Other failure points

- **Transcript quality:** The extractor receives ASR text and speaker
  attribution as produced. It cannot recover omitted or misrecognized speech.
  Segments remain separate in the prompt, so context can span segment IDs, but
  awkward boundaries or attribution errors can make relationships harder to
  interpret. Known ASR failures remain documented in
  [test status](../status/test-status.md).
- **Model reasoning:** If the full transcript fits, the LLM can still miss a
  long-distance reference, contradiction or later correction. The prompt asks
  for supported claims and citations, but reasoning quality is not guaranteed.
- **Evidence validation:** Citation checks confirm cited segment IDs exist;
  they do not prove that a paraphrased claim is semantically entailed by those
  segments. Owner and deadline checks cover specific evidence rules, not all
  factual meaning. Human review remains necessary.
- **Host resources:** Slice 3.2 measured system-wide swap use of 4.28 GiB at
  baseline and 5.12 GiB at peak during a short synthetic handoff test. Most
  swap was already in use before that run. These figures are not attributed
  solely to MeetScribe and do not predict memory use for longer recordings;
  see the [GPU handoff report](../reports/sprint-3-slice-3.2.md).

## Follow-up for slice 3.7

Continue the context investigation with the Italian and known-ASR-error cases,
and assess the generated English minutes against their references. The English
MP3 run supplies real prompt/generation counts and sampled resource data, but
not a quality result or a context-capacity guarantee. Record any duration
observations as test-specific, not as universal minute limits. Keep fail-fast
behavior for over-budget transcripts; do not silently truncate or chunk them.

For audio storage and processing-capacity estimates, see
[storage and scaling](../architecture/storage-and-scaling.md).

Exploratory chunking, conventional RAG and GraphRAG options—with tradeoffs and
a proposed evaluation sequence—are documented in
[minutes retrieval options](../architecture/minutes-retrieval-options.md). No strategy has
been selected or implemented.
