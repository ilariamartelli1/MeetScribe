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

The current context budget is **4,096 tokens**, with a maximum of **768
generated tokens** and a **128-token** chat-template reserve. The preflight
prompt bound is its UTF-8 byte length plus the template reserve; that bound and
the generation reserve must fit within 4,096. This is a conservative estimate,
not tokenizer output. The full Pydantic response schema is sent separately as
Ollama's structured-output format; a real generation is still needed to verify
the template reserve and runtime token accounting.

If the bound exceeds the context size, extraction fails before model
inference with `MinutesContextBudgetError`. No partial draft or silent
segment-dropping fallback is produced.

## Why the budget exists and how it relates to the GPU

The context window covers the prompt and generated response, not just
transcript text. Instructions, the output-shape example and serialized segment
metadata use context too. `num_predict=768` reserves bounded room for a
complete minutes response; if the model stops early or reaches its generation
limit, the extractor rejects the result rather than exporting partial JSON.
The full transcript must fit, so the app does not silently drop early or late
segments to make a request pass.

The limit also bounds inference cost. With other settings fixed, a larger
context generally needs a larger KV cache, increasing GPU or host memory use
and potentially latency. The exact cost depends on the model, quantization and
runtime. But **4,096 tokens is not a measured maximum imposed by the RTX 4050's
6 GB VRAM**. It matches the context configured for earlier smoke/handoff
checks; it was not derived from measured VRAM/RAM limits or a context-capacity
sweep. Calling it cautious means only that it is a small configured value, not
that 4,096 was demonstrated to be a safe hardware boundary. It is neither a
measured maximum nor a proven necessary minimum. CPU and future VM deployments
also need their own resource measurements.

In the 3.7 run, the **11,322-token** figure was a preflight upper bound, and
inference never started. It proves that the configured guard rejects that
input; it does **not** prove that the GPU ran out of memory or that the
tokenizer would count the prompt as 11,322 tokens. The byte-based bound may
exceed the runtime tokenizer count. Raising the budget requires measuring
actual prompt/template accounting, memory and latency, then validating output
completeness; the current fail-fast behavior remains in place.

## Observed limit

In the first approved slice 3.7 run, a **297.305-second English recording**
transcribed successfully, but its conservative total context estimate was
**11,322 tokens**, above the **4,096-token** budget. The model was not called;
this is evidence of a context-budget rejection, not an LLM quality failure.
The Italian and two-voice diagnostic recordings were not reached. No actual
generation token counts, draft-quality result, correction-time measurement
or minutes-stage resource report came from that run. See the [slice 3.7
report](sprint-3-slice-3.7.md) for the full outcome.

This does not establish a fixed maximum meeting duration. Transcript token
count varies with language, speaking rate, vocabulary and transcript content;
segment metadata and prompt instructions also consume budget. A successful
upload or transcription therefore does not guarantee that minutes extraction
fits. The current five-minute/100 MB upload limit is separate from the
summary-context limit; this 297.305-second recording transcribed successfully
but still exceeded the minutes budget.

## Other failure points

- **Transcript quality:** The extractor receives ASR text and speaker
  attribution as produced. It cannot recover omitted or misrecognized speech.
  Segments remain separate in the prompt, so context can span segment IDs, but
  awkward boundaries or attribution errors can make relationships harder to
  interpret. Known ASR failures remain documented in
  [test status](test-status.md).
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
  see the [GPU handoff report](sprint-3-slice-3.2.md).

## Follow-up for slice 3.7

When context investigation resumes, expand this note with real prompt and
generation token counts, context/template/structured-output overhead, and
quality results for representative English and Italian transcripts, including
references across segments. Record any duration observations as test-specific,
not as universal minute limits. Decide explicitly how over-budget transcripts
should be handled; until then, retain fail-fast behavior and do not silently
truncate, chunk or change models.

For audio storage and processing-capacity estimates, see
[storage and scaling](storage-and-scaling.md).

Exploratory chunking, conventional RAG and GraphRAG options—with tradeoffs and
a proposed evaluation sequence—are documented in
[minutes retrieval options](minutes-retrieval-options.md). No strategy has
been selected or implemented.
