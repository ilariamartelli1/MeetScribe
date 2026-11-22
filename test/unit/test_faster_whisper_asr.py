"""Verify faster-whisper loading, language, VAD and timestamp conversion."""

import hashlib
import struct
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.adapters import faster_whisper_asr as adapter
from app.core.asr_service import (
    InvalidTranscriptError,
    NoSpeechError,
    TranscriptionService,
)


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (tmp_path / name).write_bytes(b"test")
    digest = hashlib.sha256(b"test").hexdigest()
    monkeypatch.setattr(
        adapter,
        "MODEL_MANIFEST",
        replace(
            adapter.MODEL_MANIFEST,
            files={name: digest for name in adapter.MODEL_MANIFEST.files},
            optional_files={},
        ),
    )
    return tmp_path


def test_cuda_adapter_preserves_timestamps_and_requests_transcription(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cuda"
    model.model.compute_type = "int8_float16"
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0.1, end=0.5, text=" non ")]),
        None,
    )
    constructor = Mock(return_value=model)
    monkeypatch.setattr(adapter, "WhisperModel", constructor)
    monkeypatch.setattr(
        "app.adapters.faster_whisper_asr.ctranslate2.get_cuda_device_count", lambda: 1
    )
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [0] * 16000)

    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cuda")
    transcript = TranscriptionService(transcriber).transcribe(Path("speech.wav"), "it")

    assert transcript.duration_ms == 1000
    assert [(s.start_ms, s.end_ms, s.text) for s in transcript.segments] == [
        (100, 500, "non")
    ]
    assert transcript.provenance.device == "cuda"
    assert transcript.provenance.compute_type == "int8_float16"
    assert transcript.provenance.model_revision == adapter.MODEL_MANIFEST.revision
    assert model.transcribe.call_args.kwargs == {
        "language": "it",
        "task": "transcribe",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "chunk_length": 30,
    }
    assert constructor.call_args.kwargs["local_files_only"] is True
    assert constructor.call_args.kwargs["compute_type"] == "int8_float16"
    assert (
        transcript.provenance.beam_size
        == model.transcribe.call_args.kwargs["beam_size"]
    )
    assert (
        transcript.provenance.vad_enabled
        == model.transcribe.call_args.kwargs["vad_filter"]
    )
    assert transcript.provenance.condition_on_previous_text is False
    assert transcript.provenance.chunk_length_seconds == 30


def test_cpu_is_explicit_and_silence_is_rejected(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    constructor = Mock(return_value=model)
    monkeypatch.setattr(adapter, "WhisperModel", constructor)
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [])
    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cpu")
    assert constructor.call_args.kwargs["compute_type"] == "int8_float32"
    with pytest.raises(NoSpeechError):
        TranscriptionService(transcriber).transcribe(Path("silent.wav"))
    model.transcribe.assert_not_called()


def test_cpu_transcription_records_effective_compute_type(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0.1, end=0.5, text="hello")]),
        None,
    )
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [0] * 16000)

    transcript = TranscriptionService(
        adapter.FasterWhisperTranscriber(model_dir, "cpu")
    ).transcribe(Path("speech.wav"))

    assert transcript.provenance.device == "cpu"
    assert transcript.provenance.compute_type == "int8_float32"
    assert transcript.provenance.chunk_length_seconds == 30
    assert model.transcribe.call_args.kwargs["chunk_length"] == 30


@pytest.mark.parametrize(("left", "right"), [(10000, 0), (0, 10000)])
def test_stereo_reaches_whisper_as_mono(
    model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    left: int,
    right: int,
) -> None:
    audio_path = tmp_path / "stereo.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<hh", left, right) * 16000)
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0.1, end=0.5, text="hello")]),
        None,
    )
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    transcript = TranscriptionService(
        adapter.FasterWhisperTranscriber(model_dir, "cpu")
    ).transcribe(audio_path)
    waveform = model.transcribe.call_args.args[0]
    assert waveform.ndim == 1
    assert len(waveform) == 16000
    assert max(abs(value) for value in waveform) > 0.1
    assert transcript.duration_ms == 1000


def test_missing_gpu_and_wrong_model_are_rejected(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.adapters.faster_whisper_asr.ctranslate2.get_cuda_device_count", lambda: 0
    )
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        adapter.FasterWhisperTranscriber(model_dir, "cuda")
    (model_dir / "model.bin").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")


@pytest.mark.parametrize(
    "filename", ["config.json", "tokenizer.json", "preprocessor_config.json"]
)
def test_modified_model_metadata_is_rejected(model_dir: Path, filename: str) -> None:
    (model_dir / filename).write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")


def test_missing_weights_from_manifest_is_rejected_before_loading(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = adapter.MODEL_MANIFEST
    files = {
        name: digest for name, digest in manifest.files.items() if name != "model.bin"
    }
    monkeypatch.setattr(adapter, "MODEL_MANIFEST", replace(manifest, files=files))
    with pytest.raises(ValueError, match="manifest must pin model.bin"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")


def test_close_unloads_once_and_rejects_later_transcription(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cpu")
    transcriber.close()
    transcriber.close()
    model.model.unload_model.assert_called_once()
    with pytest.raises(RuntimeError, match="unloaded"):
        transcriber.transcribe(Path("sample.wav"), "en")


def test_missing_model_file_and_mismatched_runtime_are_rejected(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = model_dir / "config.json"
    config.unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")
    config.write_bytes(b"test")
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "float32"
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    with pytest.raises(RuntimeError, match="requested device and compute type"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")
    model.model.unload_model.assert_called_once()


def test_silence_with_no_segments_and_nonfinite_timestamps_fail(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [0] * 16000)
    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cpu")
    model.transcribe.return_value = (iter([]), None)
    with pytest.raises(NoSpeechError, match="No speech"):
        TranscriptionService(transcriber).transcribe(Path("silence.wav"))
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=float("nan"), end=0.5, text="no")]),
        None,
    )
    with pytest.raises(ValueError, match="non-finite"):
        transcriber.transcribe(Path("sample.wav"), "en")


def test_out_of_bounds_segments_are_rejected_without_clamping_or_retry(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8_float32"
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0.1, end=1.5, text="wrong")]),
        None,
    )
    monkeypatch.setattr(adapter, "WhisperModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [0] * 16000)
    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cpu")

    with pytest.raises(InvalidTranscriptError):
        TranscriptionService(transcriber).transcribe(Path("speech.mp3"))
    assert model.transcribe.call_count == 1
    assert model.transcribe.call_args.kwargs["condition_on_previous_text"] is False
