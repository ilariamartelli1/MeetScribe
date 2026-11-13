"""Compare Whisper-derived normalizers and check all candidates for safety risks."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from compare_text_normalizers import CASES, Case, canonicalize, text2num_normalizer  # type: ignore[import-not-found]
from itn.english.inverse_normalizer import InverseNormalizer as EnglishITN  # type: ignore[import-untyped]
from itn.italian.inverse_normalizer import InverseNormalizer as ItalianITN  # type: ignore[import-untyped]
from nemo_text_processing.inverse_text_normalization.inverse_normalize import (  # type: ignore[import-not-found]
    InverseNormalizer as NemoITN,
)
from nemo_text_processing.text_normalization.normalize import (  # type: ignore[import-not-found]
    Normalizer as NemoTN,
)
from tn.english.normalizer import Normalizer as EnglishTN  # type: ignore[import-untyped]
from tn.italian.normalizer import Normalizer as ItalianTN  # type: ignore[import-untyped]
from whisper_normalizer.basic import BasicTextNormalizer  # type: ignore[import-not-found]
from whisper_normalizer.english import EnglishTextNormalizer  # type: ignore[import-not-found]

SAFETY_CASES = (
    Case("en-sign-literal", "en", "sign", "-27", "27", False),
    Case("en-currency-literal", "en", "currency", "€27", "$27", False),
    Case("en-currency-missing", "en", "currency", "€27", "27", False),
    Case("en-percent", "en", "percent", "5%", "5", False),
    Case(
        "en-parentheses", "en", "negation", "release (not approved)", "release", False
    ),
    Case("en-brackets", "en", "negation", "release [not approved]", "release", False),
    Case("en-plain-negation", "en", "negation", "do not release", "do release", False),
    Case("it-sign-literal", "it", "sign", "-27", "27", False),
    Case("it-currency-literal", "it", "currency", "€27", "$27", False),
    Case("it-currency-missing", "it", "currency", "€27", "27", False),
    Case("it-percent", "it", "percent", "5%", "5", False),
    Case(
        "it-parentheses",
        "it",
        "negation",
        "rilascio (non approvato)",
        "rilascio",
        False,
    ),
    Case(
        "it-brackets", "it", "negation", "rilascio [non approvato]", "rilascio", False
    ),
    Case("it-plain-negation", "it", "negation", "non approviamo", "approviamo", False),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="meetscribe-normalizer-safety-"
    ) as directory:
        cache = Path(directory)
        normalizers: dict[str, dict[str, Callable[[str], str]]] = {
            "text2num": {lang: text2num_normalizer(lang, 3.0) for lang in ("en", "it")},
            "text2num-threshold-0": {
                lang: text2num_normalizer(lang, 0.0) for lang in ("en", "it")
            },
            "wetext-itn": {
                "en": EnglishITN(cache_dir=str(cache / "wetext-en-itn")).normalize,
                "it": ItalianITN(cache_dir=str(cache / "wetext-it-itn")).normalize,
            },
            "wetext-tn": {
                "en": EnglishTN(cache_dir=str(cache / "wetext-en-tn")).normalize,
                "it": ItalianTN(cache_dir=str(cache / "wetext-it-tn")).normalize,
            },
            "nemo-tn": {
                lang: NemoTN(
                    input_case="lower_cased",
                    lang=lang,
                    cache_dir=str(cache / f"nemo-{lang}-tn"),
                ).normalize
                for lang in ("en", "it")
            },
            "nemo-itn": {
                "en": NemoITN(
                    input_case="lower_cased",
                    lang="en",
                    cache_dir=str(cache / "nemo-en-itn"),
                ).normalize,
            },
            "whisper": {
                "en": EnglishTextNormalizer(),
                "it": BasicTextNormalizer(),
            },
        }
        counts: dict[str, Counter[str]] = {}
        results: list[dict[str, Any]] = []
        for scope, cases in (("original", CASES), ("safety", SAFETY_CASES)):
            for case in cases:
                for name, by_language in normalizers.items():
                    if scope == "original" and name != "whisper":
                        continue
                    if case.language not in by_language:
                        continue
                    key = f"{scope}:{name}"
                    counts.setdefault(key, Counter())
                    result: dict[str, Any] = {
                        "scope": scope,
                        "id": case.id,
                        "language": case.language,
                        "category": case.category,
                        "method": name,
                        "equivalent": case.equivalent,
                        "reference": case.reference,
                        "hypothesis": case.hypothesis,
                    }
                    try:
                        reference = canonicalize(
                            by_language[case.language](canonicalize(case.reference))
                        )
                        hypothesis = canonicalize(
                            by_language[case.language](canonicalize(case.hypothesis))
                        )
                    except Exception as exc:
                        result["outcome"] = "error"
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        matched = reference == hypothesis
                        result["outcome"] = (
                            "correct_match"
                            if case.equivalent and matched
                            else "missed_match"
                            if case.equivalent
                            else "false_match"
                            if matched
                            else "correct_difference"
                        )
                        result["normalized_reference"] = reference
                        result["normalized_hypothesis"] = hypothesis
                    counts[key][result["outcome"]] += 1
                    results.append(result)
        args.output.write_text(
            json.dumps(
                {"counts": counts, "results": results}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        if any(result["outcome"] == "error" for result in results):
            raise RuntimeError(f"Normalization errors were recorded in {args.output}")


if __name__ == "__main__":
    main()
