import hashlib
import math
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import ctranslate2  # type: ignore[import-untyped]
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

from app.asr_contract import Language, ModelProvenance, Transcript, TranscriptSegment

MODEL_REVISION = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8"
MODEL_SHA256 = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
SAMPLE_RATE = 16000
Device = Literal["cpu", "cuda"]


class FasterWhisperTranscriber:
    def __init__(self, model_dir: Path, device: Device) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be explicitly set to 'cuda' or 'cpu'.")
        model_dir = model_dir.resolve()
        for name in (
            "model.bin",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
        ):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(
                    f"Missing provisioned model file: {model_dir / name}"
                )
        with (model_dir / "model.bin").open("rb") as model_file:
            if hashlib.file_digest(model_file, "sha256").hexdigest() != MODEL_SHA256:
                raise ValueError("Model weights do not match the pinned SHA-256.")
        if device == "cuda" and ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("CUDA GPU unavailable; CPU fallback is disabled.")

        compute_type = "int8_float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
        if (
            self._model.model.device != device
            or self._model.model.compute_type != compute_type
        ):
            self._model.model.unload_model()
            raise RuntimeError(
                "Model did not use the requested device and compute type."
            )
        self._provenance = ModelProvenance(
            adapter="faster-whisper",
            adapter_version=version("faster-whisper"),
            model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
            model_revision=MODEL_REVISION,
            model_sha256=MODEL_SHA256,
            device=device,
            compute_type=compute_type,
            vad_enabled=True,
            beam_size=5,
        )
        self._closed = False

    def transcribe(self, audio_path: Path, language: Language) -> Transcript:
        if self._closed:
            raise RuntimeError("The transcription model has been unloaded.")
        if language not in ("en", "it"):
            raise ValueError("Recording language must be 'en' or 'it'.")
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        duration_ms = round(len(audio) * 1000 / SAMPLE_RATE)
        if not len(audio):
            return Transcript(
                language=language,
                duration_ms=0,
                segments=(),
                provenance=self._provenance,
            )

        segments, _ = self._model.transcribe(
            audio, language=language, task="transcribe", beam_size=5, vad_filter=True
        )
        result = []
        for segment in segments:
            if not math.isfinite(segment.start) or not math.isfinite(segment.end):
                raise ValueError("Model returned non-finite timestamps.")
            if not segment.text.strip():
                continue
            result.append(
                TranscriptSegment(
                    start_ms=round(segment.start * 1000),
                    end_ms=round(segment.end * 1000),
                    text=segment.text.strip(),
                )
            )
        return Transcript(
            language=language,
            duration_ms=duration_ms,
            segments=tuple(result),
            provenance=self._provenance,
        )

    def close(self) -> None:
        if not self._closed:
            self._model.model.unload_model()
            self._closed = True
