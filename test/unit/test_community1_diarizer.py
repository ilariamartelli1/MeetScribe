"""Verify Community-1 conversion, placement, and audio handling."""

import importlib
import struct
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import pytest
from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

from app import community1_diarizer as adapter
from app.diarization_contract import SpeakerTurn
from app.diarization_service import DiarizationService

ROOT = Path(__file__).resolve().parents[2]
SMOKE_AUDIO = ROOT / "test/data/smoke"


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "0")
    verify = Mock()
    monkeypatch.setattr(adapter, "verify_model", verify)
    monkeypatch.setattr(adapter, "version", lambda _: "4.0.0")
    torch = Mock()
    torch.cuda.is_available.return_value = True
    torch.serialization.safe_globals.return_value = nullcontext()
    pipeline = Mock()
    pipeline.device.type = "cuda"
    pipeline._segmentation.model.device.type = "cuda"
    pipeline._embedding.device.type = "cuda"
    pipeline_class = Mock()
    pipeline_class.from_pretrained.return_value = pipeline
    task = SimpleNamespace(
        Specifications=object(), Problem=object(), Resolution=object()
    )
    modules = {
        "torch": torch,
        "pyannote.audio": SimpleNamespace(Pipeline=pipeline_class),
        "pyannote.audio.core.task": task,
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    return torch, pipeline, verify


def _annotation(*turns: tuple[float, float, str]) -> Mock:
    annotation = Mock()
    annotation.itertracks.return_value = [
        (SimpleNamespace(start=start, end=end), None, speaker)
        for start, end, speaker in turns
    ]
    return annotation


def _write_wav(path: Path, channels: int, frames: int = 8000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * channels * frames)


def test_pipeline_returns_regular_and_exclusive_original_timeline_turns(
    backend: tuple[Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    torch, pipeline, verify = backend
    pipeline.return_value = SimpleNamespace(
        speaker_diarization=_annotation((0.2, 1.5, "A"), (1.0, 2.5, "B")),
        exclusive_speaker_diarization=_annotation((0.2, 1.5, "A"), (1.5, 2.5, "B")),
    )
    waveform = Mock()
    waveform.shape = (1, 48000)
    monkeypatch.setattr(adapter, "_decode_mono", lambda _path, _torch: waveform)

    diarizer = adapter.Community1Diarizer(Path("/model"), "cuda")
    result = DiarizationService(diarizer).diarize(Path("meeting.wav"))

    verify.assert_called_once_with(Path("/model"), adapter.MODEL_MANIFEST)
    pipeline.to.assert_called_once_with(torch.device.return_value)
    assert result.duration_ms == 3000
    assert result.turns == (SpeakerTurn(200, 1500, "A"), SpeakerTurn(1000, 2500, "B"))
    assert result.exclusive_turns == (
        SpeakerTurn(200, 1500, "A"),
        SpeakerTurn(1500, 2500, "B"),
    )
    assert result.provenance.model_id == "pyannote/speaker-diarization-community-1"
    assert result.provenance.device == "cuda"
    pipeline.assert_called_once_with({"waveform": waveform, "sample_rate": 16000})
    diarizer.close()
    diarizer.close()
    torch.cuda.empty_cache.assert_called_once()
    with pytest.raises(RuntimeError, match="unloaded"):
        diarizer.diarize(Path("meeting.wav"))


def test_explicit_cpu_and_unavailable_cuda(
    backend: tuple[Mock, Mock, Mock],
) -> None:
    torch, pipeline, _ = backend
    torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        adapter.Community1Diarizer(Path("/model"), "cuda")
    pipeline.device.type = "cpu"
    pipeline._segmentation.model.device.type = "cpu"
    pipeline._embedding.device.type = "cpu"
    diarizer = adapter.Community1Diarizer(Path("/model"), "cpu")
    diarizer.close()
    torch.cuda.empty_cache.assert_not_called()


def test_unexpected_model_device_and_telemetry_are_rejected(
    backend: tuple[Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pipeline, _ = backend
    pipeline._embedding.device.type = "cpu"
    with pytest.raises(RuntimeError, match="requested device"):
        adapter.Community1Diarizer(Path("/model"), "cuda")
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")
    with pytest.raises(RuntimeError, match="disabled telemetry"):
        adapter.Community1Diarizer(Path("/model"), "cpu")


def test_mono_decoder_rejects_surround_and_empty_audio(tmp_path: Path) -> None:
    mono = tmp_path / "mono.wav"
    surround = tmp_path / "surround.wav"
    empty = tmp_path / "empty.wav"
    _write_wav(mono, channels=1)
    _write_wav(surround, channels=4)
    _write_wav(empty, channels=1, frames=0)
    torch = Mock()
    adapter._decode_mono(mono, torch)
    torch.frombuffer.assert_called_once()
    with pytest.raises(ValueError, match="Only mono or stereo"):
        adapter._decode_mono(surround, torch)
    with pytest.raises(ValueError, match="no decodable samples"):
        adapter._decode_mono(empty, torch)


@pytest.mark.parametrize(
    ("left", "right", "expect_signal"),
    [(10000, 0, True), (0, 10000, True), (10000, -10000, False)],
    ids=["left-only", "right-only", "opposite-phase"],
)
@pytest.mark.parametrize("extension", ("wav", "mp3"))
def test_stereo_is_mixed_for_both_model_decoders(
    tmp_path: Path, left: int, right: int, expect_signal: bool, extension: str
) -> None:
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<hh", left, right) * 16000)
    audio_path = wav_path
    if extension == "mp3":
        audio_path = tmp_path / "stereo.mp3"
        with av.open(str(wav_path)) as source, av.open(str(audio_path), "w") as output:
            encoder = output.add_stream("libmp3lame", rate=16000)
            encoder.layout = "stereo"
            for frame in source.decode(audio=0):
                for packet in encoder.encode(frame):
                    output.mux(packet)
            for packet in encoder.encode(None):
                output.mux(packet)

    whisper_audio = decode_audio(str(audio_path), sampling_rate=16000)
    torch = Mock()
    adapter._decode_mono(audio_path, torch)
    pcm = bytes(torch.frombuffer.call_args.args[0])
    diarization_audio = [value / 32768 for (value,) in struct.iter_unpack("<h", pcm)]
    assert len(whisper_audio) == len(diarization_audio) == 16000
    peak = max(abs(value) for value in diarization_audio)
    if expect_signal:
        assert peak > 0.1
    else:
        assert peak < 0.01
    assert whisper_audio == pytest.approx(diarization_audio, abs=1 / 32768)


@pytest.mark.parametrize("extension", ("wav", "mp3"))
def test_mono_decoder_preserves_duration_for_supported_formats(extension: str) -> None:
    torch = Mock()
    adapter._decode_mono(SMOKE_AUDIO / f"synthetic-meeting.{extension}", torch)
    pcm = torch.frombuffer.call_args.args[0]
    samples = len(pcm) // 2
    assert abs(samples / adapter.SAMPLE_RATE_HZ - 3.531) < 0.02


def test_missing_pipeline_and_invalid_device_are_rejected(
    backend: tuple[Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    torch, _, _ = backend
    with pytest.raises(ValueError, match="explicitly set"):
        adapter.Community1Diarizer(Path("/model"), "gpu")  # type: ignore[arg-type]
    original_import = importlib.import_module

    def missing_pipeline(name: str) -> object:
        if name == "pyannote.audio":
            return SimpleNamespace(
                Pipeline=SimpleNamespace(from_pretrained=lambda _: None)
            )
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_pipeline)
    with pytest.raises(RuntimeError, match="Failed to load"):
        adapter.Community1Diarizer(Path("/model"), "cpu")
    torch.cuda.empty_cache.assert_not_called()


def test_invalid_model_output_is_not_silently_converted() -> None:
    with pytest.raises(ValueError, match="invalid speaker or timestamp"):
        adapter._turns(_annotation((float("nan"), 1.0, "A")))
