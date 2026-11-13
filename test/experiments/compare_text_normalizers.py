"""Compare candidate normalizers on labeled English and Italian text pairs.

Run with:
    uv run --locked --with text2num==3.1.0 python test/experiments/compare_text_normalizers.py
"""

import json
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from itn.english.inverse_normalizer import InverseNormalizer as EnglishITN  # type: ignore[import-untyped]
from itn.italian.inverse_normalizer import InverseNormalizer as ItalianITN  # type: ignore[import-untyped]
from text_to_num import alpha2digit  # type: ignore[import-not-found]
from tn.english.normalizer import Normalizer as EnglishTN  # type: ignore[import-untyped]
from tn.italian.normalizer import Normalizer as ItalianTN  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Case:
    id: str
    language: str
    category: str
    reference: str
    hypothesis: str
    equivalent: bool


CASES = (
    Case("en-number-27", "en", "number", "twenty seven", "27", True),
    Case("en-number-one", "en", "number", "one recording", "1 recording", True),
    Case("en-number-zero", "en", "number", "zero", "0", True),
    Case(
        "en-number-fifteen",
        "en",
        "number",
        "fifteen reviewed recordings",
        "15 reviewed recordings",
        True,
    ),
    Case("en-number-different", "en", "number", "twenty seven", "28", False),
    Case("en-decimal", "en", "number", "two point five", "2.5", True),
    Case("en-negative", "en", "number", "minus twenty seven", "-27", True),
    Case("en-sign-different", "en", "number", "minus twenty seven", "27", False),
    Case("en-date-april", "en", "date", "On fourteen April", "On 14 April", True),
    Case("en-date-may", "en", "date", "for twenty seven May", "for 27 May", True),
    Case("en-date-different", "en", "date", "on fourteen April", "on 15 April", False),
    Case(
        "en-money-spoken",
        "en",
        "currency",
        "two thousand four hundred euros",
        "2400 euros",
        True,
    ),
    Case(
        "en-money-symbol",
        "en",
        "currency",
        "two thousand four hundred euros",
        "€2,400",
        True,
    ),
    Case("en-money-punct", "en", "currency", "€2400", "€2,400", True),
    Case(
        "en-money-other-currency",
        "en",
        "currency",
        "two thousand four hundred dollars",
        "€2,400",
        False,
    ),
    Case(
        "en-money-other-amount",
        "en",
        "currency",
        "two thousand four hundred euros",
        "€2,500",
        False,
    ),
    Case(
        "en-money-decimal",
        "en",
        "currency",
        "twelve euros and fifty cents",
        "€12.50",
        True,
    ),
    Case(
        "en-negation-not",
        "en",
        "negation",
        "we will not release the feature",
        "we will release the feature",
        False,
    ),
    Case(
        "en-negation-no",
        "en",
        "negation",
        "no deadline will be changed",
        "a deadline will be changed",
        False,
    ),
    Case("en-ordinal", "en", "number", "the first review", "the 1st review", True),
    Case(
        "en-name",
        "en",
        "other",
        "Elena is taking notes",
        "Elina is taking notes",
        False,
    ),
    Case(
        "en-real-budget",
        "en",
        "fixture",
        "The estimated preparation budget is two thousand four hundred euros for the first review.",
        "The estimated preparation budget is €2,400 for the first review.",
        True,
    ),
    Case(
        "en-real-date",
        "en",
        "fixture",
        "On fourteen April, the team plans to show a short demonstration to the product group.",
        "On 14 April, the team plans to show a short demonstration to the product group.",
        True,
    ),
    Case("it-number-27", "it", "number", "ventisette", "27", True),
    Case("it-number-one", "it", "number", "un file", "1 file", True),
    Case("it-number-zero", "it", "number", "zero", "0", True),
    Case(
        "it-number-fifteen",
        "it",
        "number",
        "quindici registrazioni",
        "15 registrazioni",
        True,
    ),
    Case("it-number-different", "it", "number", "ventisette", "28", False),
    Case("it-decimal", "it", "number", "due virgola cinque", "2,5", True),
    Case("it-negative", "it", "number", "meno ventisette", "-27", True),
    Case("it-sign-different", "it", "number", "meno ventisette", "27", False),
    Case("it-date-april", "it", "date", "il quattordici aprile", "il 14 aprile", True),
    Case("it-date-may", "it", "date", "il ventisette maggio", "il 27 maggio", True),
    Case(
        "it-date-different",
        "it",
        "date",
        "il quattordici aprile",
        "il 15 aprile",
        False,
    ),
    Case(
        "it-money-spoken",
        "it",
        "currency",
        "duemila quattrocento euro",
        "2400 euro",
        True,
    ),
    Case(
        "it-money-symbol",
        "it",
        "currency",
        "duemila quattrocento euro",
        "2.400 €",
        True,
    ),
    Case("it-money-punct", "it", "currency", "2400 €", "2.400 €", True),
    Case(
        "it-money-other-currency",
        "it",
        "currency",
        "duemila quattrocento dollari",
        "2.400 €",
        False,
    ),
    Case(
        "it-money-other-amount",
        "it",
        "currency",
        "duemila quattrocento euro",
        "2.500 €",
        False,
    ),
    Case(
        "it-money-decimal",
        "it",
        "currency",
        "dodici euro e cinquanta centesimi",
        "12,50 €",
        True,
    ),
    Case(
        "it-negation-non",
        "it",
        "negation",
        "non approviamo il rilascio",
        "approviamo il rilascio",
        False,
    ),
    Case(
        "it-negation-no",
        "it",
        "negation",
        "no, non approviamo",
        "sì, approviamo",
        False,
    ),
    Case("it-cents-not-hundred", "it", "currency", "centesimi", "100o", False),
    Case("it-word", "it", "other", "deve essere riletta", "deve essere diletta", False),
    Case(
        "it-real-budget",
        "it",
        "fixture",
        "La dimostrazione è prevista per il quattordici aprile, con un budget di duemila quattrocento euro.",
        "La dimostrazione è prevista per il 14 aprile, con un budget di 2400 euro.",
        True,
    ),
    Case(
        "it-real-date",
        "it",
        "fixture",
        "Il gruppo si riunirà nuovamente il ventisette maggio per esaminare le correzioni.",
        "Il gruppo si riunirà nuovamente il 27 maggio per esaminare le correzioni.",
        True,
    ),
)


def canonicalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text2num_normalizer(language: str, threshold: float) -> Callable[[str], str]:
    def normalize(text: str) -> str:
        result: str = alpha2digit(text, language, threshold=threshold)
        return result

    return normalize


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="meetscribe-normalizers-") as directory:
        cache = Path(directory)
        normalizers: dict[str, dict[str, Callable[[str], str]]] = {
            "text2num": {lang: text2num_normalizer(lang, 3.0) for lang in ("en", "it")},
            "text2num-threshold-0": {
                lang: text2num_normalizer(lang, 0.0) for lang in ("en", "it")
            },
            "wetext-itn": {
                "en": EnglishITN(cache_dir=str(cache / "en-itn")).normalize,
                "it": ItalianITN(cache_dir=str(cache / "it-itn")).normalize,
            },
            "wetext-tn": {
                "en": EnglishTN(cache_dir=str(cache / "en-tn")).normalize,
                "it": ItalianTN(cache_dir=str(cache / "it-tn")).normalize,
            },
        }
        counts: dict[str, Counter[str]] = {name: Counter() for name in normalizers}
        results = []
        for case in CASES:
            for name, by_language in normalizers.items():
                normalize = by_language[case.language]
                reference = canonicalize(normalize(canonicalize(case.reference)))
                hypothesis = canonicalize(normalize(canonicalize(case.hypothesis)))
                matched = reference == hypothesis
                outcome = (
                    "correct_match"
                    if case.equivalent and matched
                    else "missed_match"
                    if case.equivalent
                    else "false_match"
                    if matched
                    else "correct_difference"
                )
                counts[name][outcome] += 1
                results.append(
                    {
                        "id": case.id,
                        "language": case.language,
                        "category": case.category,
                        "method": name,
                        "equivalent": case.equivalent,
                        "outcome": outcome,
                        "reference": case.reference,
                        "hypothesis": case.hypothesis,
                        "normalized_reference": reference,
                        "normalized_hypothesis": hypothesis,
                    }
                )
        print(
            json.dumps(
                {"counts": counts, "results": results}, ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    main()
