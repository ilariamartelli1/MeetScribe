"""Verify faster-whisper loading, language, VAD and timestamp conversion."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import faster_whisper_asr as adapter
from app.asr_service import NoSpeechError, TranscriptionService


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (tmp_path / name).write_bytes(b"test")
    monkeypatch.setattr(adapter, "MODEL_SHA256", hashlib.sha256(b"test").hexdigest())
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
        "app.faster_whisper_asr.ctranslate2.get_cuda_device_count", lambda: 1
    )
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [0] * 16000)

    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cuda")
    transcript = TranscriptionService(transcriber).transcribe(Path("speech.wav"), "it")

    assert transcript.duration_ms == 1000
    assert [(s.start_ms, s.end_ms, s.text) for s in transcript.segments] == [
        (100, 500, "non")
    ]
    assert transcript.provenance.device == "cuda"
    assert transcript.provenance.model_revision == adapter.MODEL_REVISION
    assert model.transcribe.call_args.kwargs == {
        "language": "it",
        "task": "transcribe",
        "beam_size": 5,
        "vad_filter": True,
    }
    assert constructor.call_args.kwargs["local_files_only"] is True


def test_cpu_is_explicit_and_silence_is_rejected(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8"
    constructor = Mock(return_value=model)
    monkeypatch.setattr(adapter, "WhisperModel", constructor)
    monkeypatch.setattr(adapter, "decode_audio", lambda *args, **kwargs: [])
    transcriber = adapter.FasterWhisperTranscriber(model_dir, "cpu")
    assert constructor.call_args.kwargs["compute_type"] == "int8"
    with pytest.raises(NoSpeechError):
        TranscriptionService(transcriber).transcribe(Path("silent.wav"))
    model.transcribe.assert_not_called()


def test_missing_gpu_and_wrong_model_are_rejected(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.faster_whisper_asr.ctranslate2.get_cuda_device_count", lambda: 0
    )
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        adapter.FasterWhisperTranscriber(model_dir, "cuda")
    (model_dir / "model.bin").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.FasterWhisperTranscriber(model_dir, "cpu")


def test_close_unloads_once_and_rejects_later_transcription(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Mock()
    model.model.device = "cpu"
    model.model.compute_type = "int8"
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
    model.model.compute_type = "int8"
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
