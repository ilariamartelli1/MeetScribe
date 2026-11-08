"""Verify default application settings and their immutability."""

from dataclasses import FrozenInstanceError

import pytest

from app.config import Settings


def test_default_settings() -> None:
    settings = Settings()

    assert settings.app_name == "MeetScribe"
    assert settings.api_prefix == "/api/v1"

    with pytest.raises(FrozenInstanceError):
        settings.api_prefix = "/other"  # type: ignore[misc]
