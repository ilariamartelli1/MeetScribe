# Meeting-minutes retrieval and context options

## Status and scope

This is an exploratory design note for the later minutes-context
investigation. It compares ways to give the summary model useful context from
long transcripts; it does not select or implement an architecture. The current
extractor still sends the complete transcript when it fits and fails before
inference when it does not. Do not silently truncate the transcript or treat
this note as approval to change that behavior.

The development host has an RTX 4050 Laptop GPU with 6 GB VRAM and about 15 GB
RAM. The future VM's hardware is unknown. Any resource comparison must be
measured on the intended target; none of the options below has established
resource requirements for that VM.

## Why consider retrieval

The current minutes prompt contains every attributed transcript segment,
including its segment ID, timestamps, speaker ID and text. Its configured
context is 4,096 tokens. In the first approved slice 3.7 run, a 297.305-second
English recording had a conservative total context estimate of 11,322 tokens,
so `MinutesContextBudgetError` stopped extraction before the model ran. This
shows a context-budget limit, not a minutes-quality failure. See the
[context-scaling report](minutes-context-scaling.md) and
[slice 3.7 report](sprint-3-slice-3.7.md).

Retrieval could select or organize useful evidence for a request without
putting the entire transcript in one prompt. But MeetScribe needs a complete
meeting summary—not only an answer to one question. A retrieval step can omit
an important but low-similarity decision, action item, correction or
qualification. It can also fail to retrieve a distant antecedent.

For example:

- Segment 1: “What about the new document draft?”
- Segment 2: “I'll send it next Friday.”

A useful action description would be “Send the new document draft,” supported
by both segments. That resolution comes from model reasoning over context; a
JSON schema and valid segment IDs do not guarantee it. Retrieval should keep
the evidence and its provenance available, but ASR errors, diarization errors,
segment boundaries and model reasoning remain separate quality risks.

## Options

### 1. Keep the full-transcript prompt as the baseline

Continue sending all segments in one request. A larger-context model or
configuration could be evaluated separately, but must be measured on the
actual target hardware rather than selected from an advertised context size.

**Pros:** Preserves the whole transcript for one model pass; simplest to
reason about and compare; avoids retrieval misses and a merge stage.

**Cons:** The complete prompt may exceed available context or resource limits.
Increasing context does not guarantee better reasoning and can increase
latency and memory use. It does not solve transcripts larger than the chosen
limit.

**Fit:** Keep as the quality baseline for recordings that fit. It is not a
solution to the observed 11,322-token case under the current 4,096-token
configuration.

### 2. Chronological, overlapping chunk-and-merge

Split the transcript into ordered, prompt-bounded windows with overlap. Extract
candidate topics, decisions and action items from each window, retaining the
original segment IDs and citations. Then merge the candidates, deduplicate
them, and reconcile later corrections against earlier statements.

**Pros:** Processes the full transcript in bounded pieces; simpler than
building a graph; preserves chronology; overlap can keep references that cross
a chunk boundary together. Each extraction can be traced to source segments.

**Cons:** A reference farther away than the overlap may still be missed.
Chunk-level summaries can lose qualifications before the merge. Multiple
inference calls add latency and token cost; duplicates and contradictions
need careful handling; the merge prompt can itself exceed its budget.

**Fit:** Strong first experiment for complete minutes because it gives every
part of the meeting an extraction pass. It still needs explicit tests for
long-distance references, repeated items, negation and later changes.

### 3. Conventional transcript RAG

Index transcript units for lexical search, semantic-vector search, or a
hybrid of both. Retrieve candidate segments for summary tasks, then include
nearby segments in timestamp order so the model can see local context. A
coverage pass or chunked first pass may still be needed to find topics and
actions that a targeted query would not retrieve.

**Pros:** Can bring distant but related segments together without sending the
whole transcript; typically less indexing complexity than a knowledge graph;
hybrid search can combine exact terms with semantic similarity.

**Cons:** Search can miss relevant segments, especially short references,
negations, corrections, names or unusual wording. A single query or fixed
top-k is not a reliable way to discover every item in a meeting. Embeddings,
query formulation and ranking add dependencies and need local-resource
measurements.

**Fit:** A useful comparison if chunk-and-merge handles coverage but performs
poorly on distant references, or if retrieval can reduce the merge context
without losing important facts.

### 4. GraphRAG

GraphRAG is a particular, more structured RAG approach. Its documented
indexing pipeline divides source text into units, uses an LLM to extract
entities and relationships, builds a graph, clusters it into communities,
and creates community summaries. At query time, its search modes use graph
structure and summaries for different kinds of questions. See the
[GraphRAG documentation](https://microsoft.github.io/graphrag/) and
[Microsoft's repository](https://github.com/microsoft/graphrag).

For MeetScribe, a graph might connect a person or speaker, a document, an
action and the transcript segments that support each relationship. Retrieval
could then gather evidence from distant parts of a meeting. Any useful graph
nodes and edges would need links to the original segment IDs; generated
summaries must not replace the source evidence.

**Pros:** Represents relationships explicitly instead of relying only on
similarity; can help find related entities or claims spread across a long
transcript; may be more useful for broad questions over a future corpus of
meetings.

**Cons:** Indexing adds LLM work, graph construction, clustering, summaries,
storage and failure modes before minutes generation. Incorrect entity or
relationship extraction can mislead every later query. A graph does not
guarantee that “it” is resolved correctly, preserve chronology by itself, or
find every action item. Community-level/global search may be unnecessary for
one meeting. Local execution and resource costs must be verified; do not
assume the framework's defaults fit MeetScribe's offline design.

As accessed on 2026-11-22, the repository describes GraphRAG as a research
project in largely maintenance mode and warns that indexing can be expensive.
Recheck that status before any adoption. GraphRAG is a research candidate
here, not a current dependency or recommended default.

**Fit:** Consider only if measured cases show that simpler approaches miss
important relationship-based context, or if a later product requirement calls
for questions across many stored meetings. Cross-meeting use would also
require separate decisions about persistent storage, permissions, retention
and data isolation; those capabilities are not implied by the current
temporary, local app.

## Cross-cutting requirements

- Keep all transcript processing, embeddings, indexes and model calls local
  under the current privacy direction. Do not send meeting data to hosted
  services for an experiment.
- Preserve segment IDs, speaker IDs and original timestamps. User-facing
  output remains minutes-only; retrieval is an internal processing step.
- Keep evidence traceable to the exact source segments. Existing citation
  checks establish that cited IDs exist, not that a generated claim is
  semantically correct. Human review remains necessary.
- Do not treat retrieval as ASR correction. Measure transcription and
  diarization quality separately from context selection and minutes quality.
- If a transcript revision changes, any derived index or graph must identify
  which revision it represents and be rebuilt or invalidated as appropriate.
  The export's `stale` field describes minutes citing a different transcript
  revision; it is not an index-freshness indicator. The current upload flow
  does not offer transcript replacement or re-transcription.

## Suggested evaluation sequence

1. Keep the current full-transcript path as a baseline on transcripts that fit.
   Record the exact transcript, model/configuration, prompt and actual token
   counts when inference runs. The 11,322-token estimate is not an actual
   model token count because inference did not start.
2. Compare an overlapping chunk-and-merge prototype against the baseline on
   human-checked references. Use the same transcript text for each method so
   ASR variation does not confound the comparison. Include far-apart
   antecedents, negation, later corrections, multiple action items, owners and
   deadlines, and English and Italian cases.
3. If the main remaining failures are retrieval-related, add a conventional
   lexical/semantic hybrid retrieval comparison. Test it for both relevant
   evidence recall and whole-meeting coverage, not just whether a selected
   example fits the context budget.
4. Evaluate GraphRAG only if those results identify relationship-heavy cases
   that simpler methods do not handle, or a concrete cross-meeting use case.
   Compare graph indexing cost and quality improvements against the simpler
   alternatives before choosing it.

For each approach, report action-item and decision precision/recall, correct
cross-segment reference resolution, chronology and correction handling,
source-citation correctness, unsupported claims and omitted items. Also
measure indexing time, model-call count, prompt/generation tokens, end-to-end
latency, host RAM, VRAM, swap and storage. Use the same offline model and
hardware for comparisons where possible, and record any difference. A method
that fits the context but drops important meeting content is not a successful
result.

## Current recommendation

Do not change the current fail-fast behavior or mark slice 3.7 complete based
on this design note. First measure a coverage-preserving chunk-and-merge
approach against the current baseline. Use conventional RAG as a next,
lower-complexity retrieval comparison if distant-context failures remain.
Treat GraphRAG as a higher-cost research option, justified only by measurable
gains on relationship-heavy or future cross-meeting tasks. No option is
selected yet.
