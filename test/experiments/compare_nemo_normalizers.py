"""Run NeMo against cases exported by compare_text_normalizers.py.

NeMo must be installed in its own Python environment; see the comparison report.
"""

import argparse
import json
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from nemo_text_processing.inverse_text_normalization.inverse_normalize import (  # type: ignore[import-not-found]
    InverseNormalizer,
)
from nemo_text_processing.text_normalization.normalize import Normalizer  # type: ignore[import-not-found]


def canonicalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    cases = [case for case in baseline["results"] if case["method"] == "text2num"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Baseline must contain one text2num result per case.")

    with tempfile.TemporaryDirectory(prefix="meetscribe-nemo-") as directory:
        cache = Path(directory)
        normalizers: dict[str, dict[str, Callable[[str], str]]] = {
            "nemo-tn": {
                lang: Normalizer(
                    input_case="lower_cased", lang=lang, cache_dir=str(cache / lang)
                ).normalize
                for lang in ("en", "it")
            },
            "nemo-itn": {
                "en": InverseNormalizer(
                    input_case="lower_cased", lang="en", cache_dir=str(cache / "en-itn")
                ).normalize,
            },
        }
        counts: dict[str, Counter[str]] = {name: Counter() for name in normalizers}
        results: list[dict[str, Any]] = []
        for case in cases:
            for name, by_language in normalizers.items():
                if case["language"] not in by_language:
                    continue
                normalize = by_language[case["language"]]
                result: dict[str, Any] = {
                    key: case[key]
                    for key in (
                        "id",
                        "language",
                        "category",
                        "equivalent",
                        "reference",
                        "hypothesis",
                    )
                }
                result["method"] = name
                try:
                    reference = canonicalize(normalize(canonicalize(case["reference"])))
                    hypothesis = canonicalize(
                        normalize(canonicalize(case["hypothesis"]))
                    )
                except Exception as exc:
                    result["outcome"] = "error"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    matched = reference == hypothesis
                    result["outcome"] = (
                        "correct_match"
                        if case["equivalent"] and matched
                        else "missed_match"
                        if case["equivalent"]
                        else "false_match"
                        if matched
                        else "correct_difference"
                    )
                    result["normalized_reference"] = reference
                    result["normalized_hypothesis"] = hypothesis
                counts[name][result["outcome"]] += 1
                results.append(result)
        args.output.write_text(
            json.dumps(
                {"counts": counts, "results": results}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
