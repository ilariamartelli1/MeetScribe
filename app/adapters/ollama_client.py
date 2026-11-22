import asyncio
import json
import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

from app.infrastructure.inference_gate import InferenceGate
from app.infrastructure.model_integrity import MANIFEST_DIR

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MANIFEST_PATH = MANIFEST_DIR / "summary_model.toml"
OLLAMA_DONE_REASON_STOP: Final = "stop"
_OLLAMA_MANIFEST = tomllib.loads(OLLAMA_MANIFEST_PATH.read_text())
MODEL_NAME = _OLLAMA_MANIFEST["model"]
MODEL_DIGEST = _OLLAMA_MANIFEST["model_digest"].removeprefix("sha256:")


class OllamaClientError(RuntimeError):
    pass


class OllamaProtocolError(OllamaClientError):
    pass


class OllamaModelUnavailableError(OllamaClientError):
    pass


class OllamaModelDigestError(OllamaClientError):
    pass


class OllamaModelReleaseError(OllamaClientError):
    pass


class OllamaModelReleaseTimeoutError(OllamaModelReleaseError):
    pass


@dataclass(frozen=True)
class OllamaGenerationResult:
    text: str
    done_reason: str
    prompt_tokens: int | None
    generated_tokens: int | None


class OllamaClient:
    """Parse Ollama's streaming ``POST /api/generate`` response structure.

    Generation event fields, including ``done_reason``, come from Ollama's API.
    The parsed generation output is returned as ``OllamaGenerationResult``.
    """

    def __init__(
        self,
        inference_gate: InferenceGate,
        *,
        unload_timeout_seconds: float = 30,
        poll_interval_seconds: float = 0.1,
        request_timeout_seconds: float = 300,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if (
            not math.isfinite(unload_timeout_seconds)
            or unload_timeout_seconds <= 0
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
            or not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("Ollama timeouts and polling interval must be positive.")
        self._inference_gate = inference_gate
        self._unload_timeout = unload_timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._closed = False
        self._client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            transport=transport,
            timeout=httpx.Timeout(
                connect=5,
                read=request_timeout_seconds,
                write=30,
                pool=5,
            ),
        )

    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        num_predict: int,
        temperature: float = 0,
        response_format: str | dict[str, object] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> OllamaGenerationResult:
        if self._closed:
            raise OllamaClientError("The Ollama client is closed.")
        if not prompt.strip():
            raise ValueError("A non-empty prompt is required.")
        if num_ctx < 1 or num_predict < 1:
            raise ValueError("Ollama context and output limits must be positive.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Ollama temperature must be finite and non-negative.")

        await self._acquire_gate()
        request_started = False
        model_released = False
        try:
            await self._verify_model()
            payload: dict[str, object] = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "keep_alive": 0,
                "think": False,
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            }
            if response_format is not None:
                payload["format"] = response_format

            request_started = True
            try:
                async with self._client.stream(
                    "POST", "/api/generate", json=payload
                ) as response:
                    response.raise_for_status()
                    result = await self._read_generation(response, on_text)
            finally:
                if request_started and not model_released:
                    try:
                        await self._wait_for_release_shielded()
                    except asyncio.CancelledError:
                        model_released = True
                        raise
                    else:
                        model_released = True
            return result
        finally:
            if request_started and not model_released:
                self._inference_gate.mark_unavailable(
                    "Ollama model release could not be verified."
                )
            self._inference_gate.release()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    async def _acquire_gate(self) -> None:
        while not self._inference_gate.try_acquire():
            await asyncio.sleep(self._poll_interval)

    async def _verify_model(self) -> None:
        response = await self._client.get("/api/tags", timeout=5)
        response.raise_for_status()
        payload = self._response_object(response, "model list")
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaProtocolError("Ollama returned an invalid model list.")
        for model in models:
            if not isinstance(model, dict):
                raise OllamaProtocolError("Ollama returned an invalid model entry.")
            if model.get("name") != MODEL_NAME:
                continue
            digest = model.get("digest")
            if not isinstance(digest, str):
                raise OllamaProtocolError("Ollama returned no model digest.")
            if digest.removeprefix("sha256:") != MODEL_DIGEST:
                raise OllamaModelDigestError(
                    "The local Ollama model digest does not match the pinned model."
                )
            return
        raise OllamaModelUnavailableError(
            f"The pinned local model {MODEL_NAME} is unavailable."
        )

    async def _read_generation(
        self,
        response: httpx.Response,
        on_text: Callable[[str], None] | None,
    ) -> OllamaGenerationResult:
        text_parts: list[str] = []
        done_reason: str | None = None
        prompt_tokens: int | None = None
        generated_tokens: int | None = None
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OllamaProtocolError(
                    "Ollama returned invalid generation data."
                ) from exc
            if not isinstance(event, dict):
                raise OllamaProtocolError(
                    "Ollama returned an invalid generation event."
                )
            if event.get("error"):
                raise OllamaClientError("Ollama reported a generation error.")
            chunk = event.get("response", "")
            if not isinstance(chunk, str):
                raise OllamaProtocolError(
                    "Ollama returned an invalid generation chunk."
                )
            if chunk:
                text_parts.append(chunk)
                if on_text is not None:
                    on_text(chunk)
            if event.get("done") is True:
                reason = event.get("done_reason")
                if not isinstance(reason, str):
                    raise OllamaProtocolError(
                        "Ollama returned no generation completion reason."
                    )
                done_reason = reason
                prompt_tokens = self._optional_int(event.get("prompt_eval_count"))
                generated_tokens = self._optional_int(event.get("eval_count"))
                break
        if done_reason is None:
            raise OllamaProtocolError("Ollama ended the response before completion.")
        return OllamaGenerationResult(
            "".join(text_parts), done_reason, prompt_tokens, generated_tokens
        )

    async def _wait_for_release_shielded(self) -> None:
        cleanup = asyncio.create_task(self._wait_until_unloaded())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        if cleanup.cancelled():
            raise OllamaModelReleaseError(
                "Ollama model release verification was cancelled."
            )
        cleanup_error = cleanup.exception()
        if cleanup_error is not None:
            raise cleanup_error
        if cancellation is not None:
            raise cancellation

    async def _wait_until_unloaded(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._unload_timeout
        while True:
            response = await self._client.get("/api/ps", timeout=5)
            response.raise_for_status()
            payload = self._response_object(response, "loaded model list")
            models = payload.get("models")
            if not isinstance(models, list):
                raise OllamaProtocolError(
                    "Ollama returned an invalid loaded model list."
                )
            for model in models:
                if not isinstance(model, dict):
                    raise OllamaProtocolError(
                        "Ollama returned an invalid loaded model entry."
                    )
                name = model.get("name")
                if not isinstance(name, str):
                    raise OllamaProtocolError(
                        "Ollama returned a loaded model without a name."
                    )
                if name == MODEL_NAME:
                    break
            else:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise OllamaModelReleaseTimeoutError(
                    "Ollama did not unload the pinned model before the timeout."
                )
            await asyncio.sleep(self._poll_interval)

    @staticmethod
    def _response_object(
        response: httpx.Response, description: str
    ) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaProtocolError(
                f"Ollama returned invalid JSON for the {description}."
            ) from exc
        if not isinstance(payload, dict):
            raise OllamaProtocolError(f"Ollama returned an invalid {description}.")
        result: dict[str, object] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                raise OllamaProtocolError(f"Ollama returned an invalid {description}.")
            result[key] = value
        return result

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None
