import math
from importlib.metadata import version
from pathlib import Path
import ctranslate2  # type: ignore[import-untyped]
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

from app.asr_contract import (
    CPU_DEVICE,
    CUDA_DEVICE,
    Device,
    Language,
    ModelProvenance,
    Transcript,
    TranscriptSegment,
)
from app.model_integrity import MANIFEST_DIR, load_manifest, verify_model

MODEL_MANIFEST = load_manifest(MANIFEST_DIR / "faster_whisper.toml")
SAMPLE_RATE = 16000
BEAM_SIZE = 5
VAD_ENABLED = True
CONDITION_ON_PREVIOUS_TEXT = False
CUDA_COMPUTE_TYPE = "int8_float16"
CPU_COMPUTE_TYPE = "int8_float32"


class FasterWhisperTranscriber:
    def __init__(self, model_dir: Path, device: Device) -> None:
        if device not in (CPU_DEVICE, CUDA_DEVICE):
            raise ValueError("Device must be explicitly set to 'cuda' or 'cpu'.")
        try:
            model_sha256 = MODEL_MANIFEST.files["model.bin"]
        except KeyError as exc:
            raise ValueError(
                f"{MODEL_MANIFEST.name} manifest must pin model.bin."
            ) from exc
        model_dir = model_dir.resolve()
        verify_model(model_dir, MODEL_MANIFEST)
        if device == CUDA_DEVICE and ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("CUDA GPU unavailable; CPU fallback is disabled.")

        compute_type = CUDA_COMPUTE_TYPE if device == CUDA_DEVICE else CPU_COMPUTE_TYPE
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
            model_id=MODEL_MANIFEST.model_id,
            model_revision=MODEL_MANIFEST.revision,
            model_sha256=model_sha256,
            device=device,
            compute_type=compute_type,
            vad_enabled=VAD_ENABLED,
            beam_size=BEAM_SIZE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
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
            audio,
            language=language,
            task="transcribe",
            beam_size=BEAM_SIZE,
            vad_filter=VAD_ENABLED,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
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
