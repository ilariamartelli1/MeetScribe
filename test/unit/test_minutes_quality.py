"""Verify the accepted deadline wording for the minutes quality diagnostic."""

import pytest

from test.integration.test_minutes_quality import (
    _deadline_matches_source_or_weekday,
)


@pytest.mark.parametrize(
    ("deadline_text", "source_phrase", "expected"),
    [
        ("Friday", "on Friday", True),
        ("friday", "on Friday", True),
        ("on Friday", "on Friday", True),
        ("Friday", "by Friday", True),
        ("by Friday", "by Friday", True),
        ("by Friday", "on Friday", False),
        ("the following Friday", "on Friday", False),
        (None, "on Friday", False),
    ],
)
def test_deadline_accepts_source_phrase_or_weekday_only(
    deadline_text: str | None, source_phrase: str, expected: bool
) -> None:
    assert _deadline_matches_source_or_weekday(deadline_text, source_phrase) is expected
