from concurrent.futures import ThreadPoolExecutor

import pytest

from app.infrastructure.inference_gate import (
    InferenceGate,
    InferenceGateUnavailableError,
)


def test_gate_serializes_work_across_threads() -> None:
    gate = InferenceGate()
    assert gate.try_acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        acquired = pool.submit(gate.try_acquire)
        assert acquired.result() is False
        pool.submit(gate.release).result()
    assert gate.try_acquire()
    gate.release()


def test_gate_stays_unavailable_after_uncertain_model_release() -> None:
    gate = InferenceGate()
    assert gate.try_acquire()
    gate.mark_unavailable("Ollama model release was not confirmed.")
    gate.release()
    with pytest.raises(
        InferenceGateUnavailableError, match="release was not confirmed"
    ):
        gate.try_acquire()


def test_gate_requires_a_reason_when_marked_unavailable() -> None:
    gate = InferenceGate()
    with pytest.raises(ValueError, match="reason is required"):
        gate.mark_unavailable("  ")
