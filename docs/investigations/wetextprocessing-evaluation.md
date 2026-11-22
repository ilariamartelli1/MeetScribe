# WeTextProcessing compatibility experiment

WeTextProcessing is pinned to Git revision
`94ab845bc83330a4819a799831462d6f7a7b93ab` in the development dependency
group, not the application runtime. The lockfile resolves Pynini 2.1.7.
The installed, non-editable source package includes English and Italian rule
data and worked on Linux x86-64 with Python 3.12.3. Pynini was installed from
a wheel; this does not establish support for other platforms.

Run the focused CPU check with:

```sh
uv sync --locked
MEETSCRIBE_TEXT_NORMALIZATION_TESTS=1 uv run --locked pytest -q test/integration/test_wetextprocessing.py
```

The check builds four finite-state grammars in a temporary directory. In one
exploratory run, building and exercising them took about 30 seconds and peaked
at approximately 534 MiB of process memory; performance depends on the system
and cache state. It is opt-in and does not run ASR.

## Observations

These are examples, not evidence that the rules are complete or suitable as
an automatic WER scoring policy:

- Inverse text normalization (ITN) converted English `twenty seven` and
  Italian `ventisette` to `27`, and converted sample negative numbers and
  dates. The example negations remained distinct from the positive sentences.
- ITN converted `two thousand four hundred euros` to `€2400` but left
  `€2,400` unchanged. Likewise, Italian `duemila quattrocento euro` became
  `2400 €`, whereas `2.400 €` remained unchanged. These representations
  therefore do not compare equal without further processing.
- Text normalization (TN) made the spoken and formatted euro examples equal
  in both languages, while distinguishing the sample dollar amounts. However,
  English `minus twenty seven` and `-27` became `minus twenty seven` and
  `negative twenty seven`, respectively; `On fourteen April` and
  `On 14 April` also did not normalize identically.

Further evaluation against the validated meeting scripts and actual ASR output
is needed before choosing TN, ITN, or additional processing for WER. Neither
the examples above nor the check establish semantic equivalence, comprehensive
negation detection, or a transcription quality threshold.
