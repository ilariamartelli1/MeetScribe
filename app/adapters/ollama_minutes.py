"""Adapt Ollama generation and its pinned settings to minutes extraction."""

import math
import tomllib

from app.adapters.ollama_client import (
    OLLAMA_DONE_REASON_STOP,
    OLLAMA_MANIFEST_PATH,
    OllamaClient,
)
from app.core.minutes_extractor import (
    BoundedMinutesExtractor,
    GenerationResult,
    MinutesExtractionBudget,
    prompt_token_upper_bound,
)


def _manifest_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Summary model manifest needs an {key!r} table.")
    return value


def _manifest_int(
    data: dict[str, object], key: str, *, allow_zero: bool = False
) -> int:
    value = data.get(key)
    if type(value) is not int or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Summary model manifest {key!r} must be a positive integer.")
    return value


def load_extraction_budget() -> MinutesExtractionBudget:
    manifest = tomllib.loads(OLLAMA_MANIFEST_PATH.read_text())
    extraction = _manifest_table(manifest, "extraction")
    smoke = _manifest_table(manifest, "smoke")
    context_size = _manifest_int(extraction, "context_size")
    measured_context_size = _manifest_int(smoke, "context_size")
    if context_size > measured_context_size:
        raise ValueError(
            "Summary extraction context cannot exceed the measured model context."
        )
    max_generated_tokens = _manifest_int(extraction, "max_generated_tokens")
    template_reserve_tokens = _manifest_int(
        extraction, "chat_template_reserve_tokens", allow_zero=True
    )
    temperature = extraction.get("temperature")
    if (
        not isinstance(temperature, (float, int))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError(
            "Summary extraction temperature must be finite and non-negative."
        )
    return MinutesExtractionBudget(
        context_size=context_size,
        max_generated_tokens=max_generated_tokens,
        chat_template_reserve_tokens=template_reserve_tokens,
        temperature=float(temperature),
    )


DEFAULT_MINUTES_EXTRACTION_BUDGET = load_extraction_budget()


class OllamaMinutesModel:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    async def generate(
        self,
        prompt: str,
        *,
        context_size: int,
        max_output_tokens: int,
        temperature: float,
        response_schema: dict[str, object],
    ) -> GenerationResult:
        result = await self._client.generate(
            prompt,
            num_ctx=context_size,
            num_predict=max_output_tokens,
            temperature=temperature,
            response_format=response_schema,
        )
        return GenerationResult(
            text=result.text,
            complete=result.done_reason == OLLAMA_DONE_REASON_STOP,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
        )


def create_ollama_minutes_extractor(client: OllamaClient) -> BoundedMinutesExtractor:
    return BoundedMinutesExtractor(
        OllamaMinutesModel(client),
        DEFAULT_MINUTES_EXTRACTION_BUDGET,
        prompt_token_upper_bound,
    )
