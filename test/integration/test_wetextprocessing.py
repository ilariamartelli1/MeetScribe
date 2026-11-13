"""Check the pinned WeTextProcessing source and its English/Italian rule data.

Set MEETSCRIBE_TEXT_NORMALIZATION_TESTS=1 to run this CPU-intensive check.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_TEXT_NORMALIZATION_TESTS") != "1",
    reason="Set MEETSCRIBE_TEXT_NORMALIZATION_TESTS=1 to check WeTextProcessing.",
)


def test_english_and_italian_rule_data(tmp_path: Path) -> None:
    from itn.english.inverse_normalizer import InverseNormalizer as EnglishITN  # type: ignore[import-untyped]
    from itn.italian.inverse_normalizer import InverseNormalizer as ItalianITN  # type: ignore[import-untyped]
    from tn.english.normalizer import Normalizer as EnglishTN  # type: ignore[import-untyped]
    from tn.italian.normalizer import Normalizer as ItalianTN  # type: ignore[import-untyped]

    en_itn = EnglishITN(cache_dir=str(tmp_path / "en-itn"))
    assert en_itn.normalize("twenty seven") == "27"
    assert en_itn.normalize("minus twenty seven") == "-27"
    assert en_itn.normalize("On fourteen April") == "On 14 April"
    assert en_itn.normalize("two thousand four hundred euros") == "€2400"
    assert en_itn.normalize("two thousand four hundred dollars") == "$2400"
    assert en_itn.normalize("we will not release") != en_itn.normalize(
        "we will release"
    )

    it_itn = ItalianITN(cache_dir=str(tmp_path / "it-itn"))
    assert it_itn.normalize("ventisette") == "27"
    assert it_itn.normalize("meno ventisette") == "-27"
    assert it_itn.normalize("il quattordici aprile") == "il 14 aprile"
    assert it_itn.normalize("duemila quattrocento euro") == "2400 €"
    assert it_itn.normalize("duemila quattrocento dollari") == "2400 $"
    assert it_itn.normalize("non approviamo") != it_itn.normalize("approviamo")

    en_tn = EnglishTN(cache_dir=str(tmp_path / "en-tn"))
    assert en_tn.normalize("two thousand four hundred euros") == en_tn.normalize(
        "€2,400"
    )
    assert en_tn.normalize("two thousand four hundred dollars") != en_tn.normalize(
        "€2,400"
    )
    it_tn = ItalianTN(cache_dir=str(tmp_path / "it-tn"))
    assert it_tn.normalize("duemila quattrocento euro") == it_tn.normalize("2.400 €")
    assert it_tn.normalize("duemila quattrocento dollari") != it_tn.normalize("2.400 €")
