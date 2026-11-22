import threading


class InferenceGateUnavailableError(RuntimeError):
    pass


class InferenceGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._unavailable_reason: str | None = None

    def try_acquire(self) -> bool:
        with self._state_lock:
            if self._unavailable_reason is not None:
                raise InferenceGateUnavailableError(
                    f"GPU inference is unavailable: {self._unavailable_reason}"
                )
            return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    def mark_unavailable(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("An unavailable reason is required.")
        with self._state_lock:
            if self._unavailable_reason is None:
                self._unavailable_reason = reason
