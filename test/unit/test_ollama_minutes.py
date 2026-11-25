"""Verify the Ollama-to-minutes boundary and pinned extraction settings."""

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from app.adapters import ollama_minutes
from app.adapters.ollama_client import OllamaClient, OllamaGenerationResult
from app.core.minutes_extractor import GenerationResult


@pytest.mark.parametrize(("reason", "complete"), [("stop", True), ("length", False)])
def test_adapter_translates_request_and_completion(reason: str, complete: bool) -> None:
    client = create_autospec(OllamaClient, instance=True)
    client.generate.return_value = OllamaGenerationResult("text", reason, 40, 20)
    schema: dict[str, object] = {"type": "object"}

    result = asyncio.run(
        ollama_minutes.OllamaMinutesModel(client).generate(
            "prompt",
            context_size=4096,
            max_output_tokens=768,
            temperature=0,
            response_schema=schema,
        )
    )

    assert result == GenerationResult("text", complete, 40, 20, reason)
    client.generate.assert_awaited_once_with(
        "prompt", num_ctx=4096, num_predict=768, temperature=0, response_format=schema
    )


def test_pinned_extraction_budget_matches_manifest() -> None:
    budget = ollama_minutes.load_extraction_budget()
    assert budget.context_size == 26112
    assert budget.max_generated_tokens == 4096
    assert budget.chat_template_reserve_tokens == 128


def test_manifest_rejects_context_larger_than_smoke_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "[extraction]\ncontext_size = 4097\n[smoke]\ncontext_size = 4096\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ollama_minutes, "OLLAMA_MANIFEST_PATH", manifest)
    with pytest.raises(ValueError, match="cannot exceed"):
        ollama_minutes.load_extraction_budget()
