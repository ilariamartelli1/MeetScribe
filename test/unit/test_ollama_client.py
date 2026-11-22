import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.inference_gate import InferenceGate, InferenceGateUnavailableError
from app.ollama_client import (
    MODEL_DIGEST,
    MODEL_NAME,
    OllamaClient,
    OllamaModelDigestError,
    OllamaModelReleaseError,
    OllamaModelUnavailableError,
    OllamaProtocolError,
)


def _tags(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"models": [{"name": MODEL_NAME, "digest": MODEL_DIGEST, "size": 1234}]},
        request=request,
    )


def _generation(text: str = "draft") -> bytes:
    return (
        json.dumps({"response": text, "done": False}).encode()
        + b"\n"
        + json.dumps(
            {
                "response": "",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
            }
        ).encode()
        + b"\n"
    )


def test_generate_uses_pinned_model_disables_retention_and_waits_for_unload() -> None:
    ps_requests = 0
    request_payload: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal ps_requests
        if request.url.path == "/api/tags":
            return _tags(request)
        if request.url.path == "/api/generate":
            request_payload.update(json.loads(request.content))
            return httpx.Response(200, content=_generation(), request=request)
        if request.url.path == "/api/ps":
            ps_requests += 1
            models = [{"name": MODEL_NAME}] if ps_requests == 1 else []
            return httpx.Response(200, json={"models": models}, request=request)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    gate = InferenceGate()
    client = OllamaClient(
        gate,
        transport=httpx.MockTransport(handle),
        poll_interval_seconds=0.001,
    )

    async def run() -> None:
        result = await client.generate(
            "Create a short draft.",
            num_ctx=4096,
            num_predict=32,
            response_format="json",
        )
        assert result.text == "draft"
        assert result.done_reason == "stop"
        assert result.prompt_tokens == 12
        assert result.generated_tokens == 3
        assert ps_requests == 2
        assert gate.try_acquire()
        gate.release()
        await client.aclose()

    asyncio.run(run())
    assert request_payload == {
        "model": MODEL_NAME,
        "prompt": "Create a short draft.",
        "stream": True,
        "keep_alive": 0,
        "think": False,
        "options": {"num_ctx": 4096, "num_predict": 32, "temperature": 0},
        "format": "json",
    }


@pytest.mark.parametrize(
    ("models", "error"),
    [
        ([], OllamaModelUnavailableError),
        ([{"name": MODEL_NAME, "digest": "wrong"}], OllamaModelDigestError),
    ],
)
def test_generation_rejects_unavailable_or_unpinned_model(
    models: list[dict[str, object]], error: type[Exception]
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": models},
                request=request,
            )
        raise AssertionError("Generation must not start for an unavailable model.")

    gate = InferenceGate()
    client = OllamaClient(
        gate,
        transport=httpx.MockTransport(handle),
        poll_interval_seconds=0.001,
    )

    async def run() -> None:
        with pytest.raises(error):
            await client.generate("Create a draft.", num_ctx=4096, num_predict=32)
        assert gate.try_acquire()
        gate.release()
        await client.aclose()

    asyncio.run(run())


def test_concurrent_generations_are_serialized_through_model_release() -> None:
    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return _tags(request)
            if request.url.path == "/api/generate":
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                return httpx.Response(200, content=_generation(), request=request)
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": []}, request=request)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    gate = InferenceGate()
    transport = TrackingTransport()
    client = OllamaClient(gate, transport=transport, poll_interval_seconds=0.001)

    async def run() -> None:
        results = await asyncio.gather(
            client.generate("First draft.", num_ctx=4096, num_predict=32),
            client.generate("Second draft.", num_ctx=4096, num_predict=32),
        )
        assert [result.text for result in results] == ["draft", "draft"]
        assert transport.maximum_active == 1
        assert gate.try_acquire()
        gate.release()
        await client.aclose()

    asyncio.run(run())


def test_cancellation_closes_request_and_waits_for_model_release() -> None:
    class HangingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"response":"part","done":false}\n'
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed = True

    class CancellationTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.stream = HangingStream()
            self.ps_requests = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return _tags(request)
            if request.url.path == "/api/generate":
                return httpx.Response(200, stream=self.stream, request=request)
            if request.url.path == "/api/ps":
                self.ps_requests += 1
                models = [{"name": MODEL_NAME}] if self.ps_requests == 1 else []
                return httpx.Response(200, json={"models": models}, request=request)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    gate = InferenceGate()
    transport = CancellationTransport()
    client = OllamaClient(gate, transport=transport, poll_interval_seconds=0.001)
    first_chunk = asyncio.Event()

    async def run() -> None:
        task = asyncio.create_task(
            client.generate(
                "Write a long draft.",
                num_ctx=4096,
                num_predict=4096,
                on_text=lambda _chunk: first_chunk.set(),
            )
        )
        await asyncio.wait_for(first_chunk.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.stream.closed
        assert transport.ps_requests == 2
        assert gate.try_acquire()
        gate.release()
        await client.aclose()

    asyncio.run(run())


def test_repeated_cancellation_cannot_skip_model_release_verification() -> None:
    class HangingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"response":"part","done":false}\n'
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            return None

    class DelayedReleaseTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.release_check_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return _tags(request)
            if request.url.path == "/api/generate":
                return httpx.Response(200, stream=HangingStream(), request=request)
            if request.url.path == "/api/ps":
                self.release_check_started.set()
                await self.allow_release.wait()
                return httpx.Response(200, json={"models": []}, request=request)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    gate = InferenceGate()
    transport = DelayedReleaseTransport()
    client = OllamaClient(gate, transport=transport, poll_interval_seconds=0.001)
    first_chunk = asyncio.Event()

    async def run() -> None:
        task = asyncio.create_task(
            client.generate(
                "Write a long draft.",
                num_ctx=4096,
                num_predict=4096,
                on_text=lambda _chunk: first_chunk.set(),
            )
        )
        try:
            await asyncio.wait_for(first_chunk.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(transport.release_check_started.wait(), timeout=1)
            task.cancel()
            assert not task.done()
            assert gate.try_acquire() is False
            transport.allow_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert gate.try_acquire()
            gate.release()
        finally:
            transport.allow_release.set()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await client.aclose()

    asyncio.run(run())


def test_unconfirmed_model_release_blocks_all_later_gpu_work() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags(request)
        if request.url.path == "/api/generate":
            return httpx.Response(200, content=_generation(), request=request)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": MODEL_NAME}]})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    gate = InferenceGate()
    client = OllamaClient(
        gate,
        transport=httpx.MockTransport(handle),
        unload_timeout_seconds=0.01,
        poll_interval_seconds=0.002,
    )

    async def run() -> None:
        with pytest.raises(OllamaModelReleaseError):
            await client.generate("Create a draft.", num_ctx=4096, num_predict=32)
        with pytest.raises(InferenceGateUnavailableError):
            gate.try_acquire()
        await client.aclose()

    asyncio.run(run())


def test_invalid_or_incomplete_ollama_responses_are_reported() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags(request)
        if request.url.path == "/api/generate":
            return httpx.Response(
                200,
                content=b'{"response":"partial","done":false}\n',
                request=request,
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []}, request=request)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = OllamaClient(
        InferenceGate(),
        transport=httpx.MockTransport(handle),
        poll_interval_seconds=0.001,
    )

    async def run() -> None:
        with pytest.raises(OllamaProtocolError, match="before completion"):
            await client.generate("Create a draft.", num_ctx=4096, num_predict=32)
        await client.aclose()

    asyncio.run(run())


def test_client_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        OllamaClient(InferenceGate(), unload_timeout_seconds=0)

    client = OllamaClient(InferenceGate())

    async def run() -> None:
        with pytest.raises(ValueError, match="non-empty prompt"):
            await client.generate(" ", num_ctx=4096, num_predict=32)
        with pytest.raises(ValueError, match="limits must be positive"):
            await client.generate("Draft.", num_ctx=0, num_predict=32)
        with pytest.raises(ValueError, match="finite and non-negative"):
            await client.generate(
                "Draft.", num_ctx=4096, num_predict=32, temperature=float("nan")
            )
        await client.aclose()

    asyncio.run(run())
