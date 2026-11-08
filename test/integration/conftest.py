"""Provide pytest fixtures to integration tests.

Pytest discovers conftest.py automatically. Its session-scoped ASR image fixture
builds the application image once per test run and shares it with ASR tests.
"""

import pytest

from test.src.asr_image import build_asr_image


@pytest.fixture(scope="session")
def asr_image() -> str:
    return build_asr_image()
