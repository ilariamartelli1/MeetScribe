"""Run the pinned Community-1 model on mono or mixed stereo audio."""

import gc
import importlib
import io
import math
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import av

from app.core.asr_contract import CPU_DEVICE, CUDA_DEVICE, Device
from app.core.diarization_contract import (
    Diarization,
    DiarizationProvenance,
    SpeakerTurn,
)
from app.infrastructure.model_integrity import MANIFEST_DIR, load_manifest, verify_model

MODEL_MANIFEST = load_manifest(MANIFEST_DIR / "community1.toml")
SAMPLE_RATE_HZ = 16_000
MILLISECONDS_PER_SECOND = 1000
PCM_INT16_SCALE = 32768


def _decode_mono(audio_path: Path, torch: Any) -> Any:
    buffer = io.BytesIO()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    sample_rate = None
    channel_layout = None
    with av.open(str(audio_path), mode="r") as container:
        if len(container.streams) != 1 or len(container.streams.audio) != 1:
            raise ValueError("Exactly one audio stream is required for diarization.")
        for frame in container.decode(container.streams.audio[0]):
            if len(frame.layout.channels) not in (1, 2):
                raise ValueError(
                    "Only mono or stereo audio is supported for diarization."
                )
            if channel_layout is not None and frame.layout.name != channel_layout:
                raise ValueError("Changing audio channel layout is unsupported.")
            channel_layout = frame.layout.name
            if not frame.sample_rate or (
                sample_rate is not None and frame.sample_rate != sample_rate
            ):
                raise ValueError(
                    "Changing or missing audio sample rate is unsupported."
                )
            sample_rate = frame.sample_rate
            for resampled in resampler.resample(frame):
                buffer.write(resampled.to_ndarray().tobytes())
        for resampled in resampler.resample(None):
            buffer.write(resampled.to_ndarray().tobytes())
    if buffer.tell() == 0:
        raise ValueError("Audio contains no decodable samples.")
    return (
        torch.frombuffer(buffer.getbuffer(), dtype=torch.int16)
        .to(dtype=torch.float32)
        .div_(PCM_INT16_SCALE)
        .unsqueeze(0)
    )


def _turns(annotation: Any) -> tuple[SpeakerTurn, ...]:
    result = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        if (
            not isinstance(speaker, str)
            or not math.isfinite(segment.start)
            or not math.isfinite(segment.end)
        ):
            raise ValueError("Diarization returned an invalid speaker or timestamp.")
        result.append(
            SpeakerTurn(
                start_ms=round(segment.start * MILLISECONDS_PER_SECOND),
                end_ms=round(segment.end * MILLISECONDS_PER_SECOND),
                speaker_id=speaker,
            )
        )
    return tuple(result)


class Community1Diarizer:
    def __init__(self, model_dir: Path, device: Device) -> None:
        if device not in (CPU_DEVICE, CUDA_DEVICE):
            raise ValueError("Device must be explicitly set to 'cuda' or 'cpu'.")
        if (
            os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("HF_HUB_DISABLE_TELEMETRY") != "1"
            or os.environ.get("PYANNOTE_METRICS_ENABLED") != "0"
        ):
            raise RuntimeError("Offline mode and disabled telemetry are required.")
        model_dir = model_dir.resolve()
        verify_model(model_dir, MODEL_MANIFEST)
        self._torch = importlib.import_module("torch")
        if device == CUDA_DEVICE and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable; CPU fallback is disabled.")
        pipeline_class = importlib.import_module("pyannote.audio").Pipeline
        task = importlib.import_module("pyannote.audio.core.task")
        with self._torch.serialization.safe_globals(
            [task.Specifications, task.Problem, task.Resolution]
        ):
            pipeline = pipeline_class.from_pretrained(model_dir)
        if pipeline is None:
            raise RuntimeError("Failed to load local Community-1 pipeline.")
        pipeline.to(self._torch.device(device))
        if (
            pipeline.device.type != device
            or pipeline._segmentation.model.device.type != device
            or pipeline._embedding.device.type != device
        ):
            raise RuntimeError("Diarization did not use the requested device.")
        self._pipeline = pipeline
        self._device = device
        self._provenance = DiarizationProvenance(
            adapter="pyannote.audio",
            adapter_version=version("pyannote-audio"),
            model_id=MODEL_MANIFEST.model_id,
            model_revision=MODEL_MANIFEST.revision,
            device=device,
        )

    def diarize(self, audio_path: Path) -> Diarization:
        if self._pipeline is None:
            raise RuntimeError("The diarization model has been unloaded.")
        waveform = _decode_mono(audio_path, self._torch)
        duration_ms = round(
            waveform.shape[-1] * MILLISECONDS_PER_SECOND / SAMPLE_RATE_HZ
        )
        output = self._pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE_HZ})
        return Diarization(
            duration_ms=duration_ms,
            turns=_turns(output.speaker_diarization),
            exclusive_turns=_turns(output.exclusive_speaker_diarization),
            provenance=self._provenance,
        )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline = None
            gc.collect()
            if self._device == CUDA_DEVICE:
                self._torch.cuda.empty_cache()
