# Text normalization comparison

This experiment compares candidates for a future transcript evaluation
pipeline. It does not calculate WER or set a pass threshold.

## Setup

- WeTextProcessing `1.3.1.dev4` at Git revision
  `94ab845bc83330a4819a799831462d6f7a7b93ab`, with Pynini 2.1.7.
- text2num 3.1.0, tested with its default threshold of 3 and separately with
  threshold 0. It is installed only for the experiment, not added to the
  project's dependencies.
- Linux x86-64, Python 3.12.3. All normalization runs locally on the CPU.

To reproduce:

```sh
uv sync --locked
uv run --locked --with text2num==3.1.0 python test/experiments/compare_text_normalizers.py > comparison.json
```

The script builds WeTextProcessing's grammars in a temporary directory and
prints JSON with the input and normalized text, expected relationship and
outcome for every case. The one-off `--with` dependency requires package
access unless already cached. Standard output contains JSON; WeTextProcessing
also writes diagnostic messages to standard error.

## Method

There are 46 hand-labeled sentence or phrase pairs: 23 English and 23 Italian,
with 29 expected matches and 17 expected differences overall. Four pairs use
the wording of the validated scripts and previously captured ASR output.
The rest are deliberately selected examples of numbers, dates, currency,
negative values, negations and other word differences; one `centesimi` versus
`100o` pair specifically tests an observed text2num transformation.

For each method, both inputs are Unicode NFKC-normalized, case-folded and
whitespace-collapsed before and after the library call. We compare the
resulting strings exactly: the common comparison code does **not** remove
punctuation, symbols, or signs, although an individual normalizer may remove
them internally. This deliberately does not pretend to be a complete WER
preprocessing pipeline. Applying different punctuation or currency rules can
change results and may conceal differences such as negative signs.

| Method | Equivalent pairs matched / 29 | Different pairs kept different / 17 | False matches |
| --- | ---: | ---: | ---: |
| text2num, default threshold | 15 | 16 | 1 |
| text2num, threshold 0 | 20 | 16 | 1 |
| WeTextProcessing inverse normalization (ITN) | 18 | 17 | 0 |
| WeTextProcessing text normalization (TN) | 21 | 17 | 0 |

These are results on a small, deliberately chosen set, not population-level
accuracy estimates. A missed match is not necessarily a meaningful WER error:
word-level scoring and punctuation handling may behave differently. Conversely,
the absence of false matches in this set does not establish semantic safety.

## Representative findings

- Both text2num configurations converted `centesimi` to `100o`, so that
  string falsely matched the intentionally constructed `100o` input. In the
  longer Italian example, `dodici euro e cinquanta centesimi` became
  `12 euro e 50 100o`. This is not a claim that `100o` occurred in our ASR
  captures.
- The default text2num threshold left `zero` and `one recording` as words;
  threshold 0 matched them against `0` and `1 recording`. But threshold 0
  also converted `un budget` to `1 budget` in an actual fixture sentence:
  treating every small-number-looking word as a number can change an
  indefinite article into an explicit count.
- Both text2num settings converted `minus twenty seven` to `minus 27`, which
  did not match `-27`. WeTextProcessing ITN matched this pair, while its TN
  produced `minus twenty seven` versus `negative twenty seven`.
- For the English budget fixture, only WeTextProcessing TN matched
  `two thousand four hundred euros` with `€2,400`. For the Italian fixture
  with `duemila quattrocento euro` versus `2400 euro`, text2num and
  WeTextProcessing TN matched, while ITN produced `2400 €` versus `2400 euro`.
  Text2num matched `two thousand four hundred euros` with `2400 euros`,
  while neither WeTextProcessing direction matched that particular pair.
- Both text2num settings and WeTextProcessing ITN matched the English
  fixture date `fourteen April` with `14 April`; WeTextProcessing TN rendered
  the latter as `the fourteenth of april`, so the full strings differed.
- All four methods kept the selected negation, changed-amount,
  changed-currency and sign examples different. The test is too small to
  prove they always do so.

In a separate small process that imported text2num and normalized one English
and one Italian amount, the observed startup plus processing was 0.02 seconds
and peak RSS was about 12 MiB. The earlier WeTextProcessing experiment built
and exercised four grammars in about 30 seconds with a peak of about 534 MiB;
that is a different workload, not a controlled performance benchmark.

## Interpretation

WeTextProcessing covers more of these examples without additional currency
logic, but neither direction is consistently best. text2num is much lighter
and useful for spoken numbers, yet it needs additional currency and
signed-number handling and has a concerning Italian `centesimi` result in
this version. Do not select an automatic scoring policy based on these totals
alone. Test candidate combinations and their error cases against the full
validated scripts and actual ASR output before choosing one.

## NeMo Text Processing

NeMo Text Processing 1.2.0 was tested against the same 46 pairs in a
separate Python 3.12 environment. The installed package includes English
and Italian TN, but **not Italian ITN**. English ITN was therefore measured
on the 23 English pairs only.

| Method | Equivalent pairs matched | Different pairs kept different | False matches |
| --- | ---: | ---: | ---: |
| NeMo TN, English and Italian | 22/29 | 17/17 | 0 |
| NeMo ITN, English only | 9/15 | 8/8 | 0 |

These are exact-string comparisons, not WER scores. The earlier totals for
WeTextProcessing and text2num above include both languages and cannot be
compared directly to the English-only NeMo ITN result.

- NeMo TN and WeTextProcessing TN produced the same match/difference outcomes
  for 45 of the 46 pairs. For the other, NeMo matched `minus twenty seven`
  with `-27`; WeTextProcessing rendered the latter `negative twenty seven`.
- Like WeTextProcessing TN, NeMo TN matched both actual budget examples but
  missed the English `fourteen April` versus `14 April` date example.
- NeMo English ITN and WeTextProcessing English ITN had the same
  match/difference outcome on all 23 English pairs. This does not imply
  their normalized strings are identical.
- No NeMo cases raised exceptions in this experiment. Its English ITN
  documentation expects lowercase input without punctuation other than
  apostrophes and dashes. This experiment retained punctuation and currency
  symbols for consistency and safety, so those results exercise inputs outside
  that documented contract and require further investigation.

The published NeMo package requires Pynini `2.1.6.post1`; the project's
current WeTextProcessing lock resolves Pynini `2.1.7`. In an isolated
environment, installing both packages together resolved to `2.1.6.post1`.
Both experiment scripts ran successfully there, and every WeTextProcessing
and NeMo result was identical to its respective earlier run. This checks the
sample cases only; no project dependency or lockfile was changed for NeMo.
The NeMo installation built `cdifflib` from source because a compatible wheel
was not available. A compatible Pynini wheel was available.

In one run, building three NeMo grammars and evaluating the cases took about
38 seconds and peaked at about 426 MiB RSS. The earlier WeTextProcessing
measurement used four grammars, so the two measurements are not a controlled
performance comparison.

To reproduce NeMo separately after generating `comparison.json` as above:

```sh
uv venv --python 3.12 /tmp/meetscribe-nemo-venv
uv pip install --python /tmp/meetscribe-nemo-venv/bin/python 'nemo-text-processing==1.2.0'
/tmp/meetscribe-nemo-venv/bin/python test/experiments/compare_nemo_normalizers.py \
  --baseline comparison.json --output nemo-comparison.json
```

The NeMo script records normalization exceptions per case in its JSON output.
The install may require a C compiler and network access. The NeMo version is
pinned in the command, but its transitive dependencies are not locked by it.

## Whisper-derived normalizers and final comparison

The additional candidate is the third-party `whisper-normalizer` 1.0.1
package, derived from OpenAI's Whisper normalization code. This is not the
official `openai-whisper` package, and identical behavior across versions has
not been established. We used its default `EnglishTextNormalizer` for English
and default `BasicTextNormalizer` for Italian. The basic normalizer does not
convert Italian number words.

The original 46-pair comparison is unchanged. We also added 14 deliberately
selected **different-meaning safety pairs**: seven per language covering
literal negative signs, currency symbols, percentages, bracketed or
parenthesized negations, and plain negations. These are separate from the
original set and were added after identifying Whisper's symbol and bracket
handling; they are not a representative sample.

| Candidate | Original equivalent pairs matched / 29 | Original differences kept / 17 | Additional safety differences kept / 14 |
| --- | ---: | ---: | ---: |
| NeMo TN | 22 | 17 | 14 |
| WeTextProcessing TN | 21 | 17 | 14 |
| WeTextProcessing ITN | 18 | 17 | 14 |
| text2num, default | 15 | 16 | 14 |
| text2num, threshold 0 | 20 | 16 | 14 |
| Whisper English / Basic Italian | 14 | 17 | 5 |

NeMo ITN was measured only in English: it matched 9/15 equivalent pairs,
kept 8/8 original different pairs different, and kept all 7 English safety
pairs different. It has no Italian ITN in the tested package.

- On the original set, Whisper's English normalizer matched 14/15 equivalent
  English pairs, including the English date and budget fixtures. Its Italian
  basic normalizer matched 0/14 equivalent Italian pairs. The aggregate score
  obscures this important language difference.
- On the safety cases, Whisper collapsed `-27` and `27` in both languages;
  its Italian basic normalizer also collapsed `€27`, `$27` and `27`, and
  `5%` and `5`. Both normalizers removed the contents of the sample brackets
  and parentheses, including `not approved` / `non approvato`. The selected
  plain negations remained distinct.
- The other methods kept all 14 additional safety pairs different, except
  that NeMo ITN was tested only on the seven English pairs. This narrow result
  is not proof that they always preserve meaning. The original, deliberately
  constructed Italian `centesimi` / `100o` false match for text2num still
  applies.

The candidates apply different transformations internally: notably, Whisper
removes punctuation that other methods retain. As such, exact matches can
reflect punctuation removal rather than better number normalization. None of
these comparisons are WER scores, and the counts alone do not establish the
best end-to-end pipeline.

On the current evidence, **WeTextProcessing is the more practical starting
candidate**: the tested source supports both directions in both project
languages and is already pinned for development. NeMo TN is also promising
and matched one more pair here, but the tested package lacks Italian ITN and
requires a different Pynini version from the current lock. The default Whisper
normalizers should not be used unmodified for meaning-sensitive evaluation
because of the observed information loss. Evaluate the remaining candidates
with JiWER on full validated scripts and ASR outputs, retaining separate
checks for critical differences, before selecting a production scoring policy.

To reproduce the Whisper and safety checks in a separate environment:

```sh
uv venv --python 3.12 /tmp/meetscribe-comparison-venv
uv pip install --python /tmp/meetscribe-comparison-venv/bin/python \
  'nemo-text-processing==1.2.0' \
  'WeTextProcessing @ git+https://github.com/wenet-e2e/WeTextProcessing.git@94ab845bc83330a4819a799831462d6f7a7b93ab' \
  'text2num==3.1.0' 'whisper-normalizer==1.0.1'
/tmp/meetscribe-comparison-venv/bin/python \
  test/experiments/compare_whisper_normalizers.py --output whisper-comparison.json
```

In the tested environment the resolver selected Pynini `2.1.6.post1`; other
transitive dependencies are not locked by this command. The experiment records
all normalized outputs and failures in JSON, and exits with an error if a
normalizer raises an exception. Building all seven grammars and running the
checks took about 84 seconds and peaked at about 631 MiB RSS in one run.
This is not a controlled comparison of individual libraries' performance.
