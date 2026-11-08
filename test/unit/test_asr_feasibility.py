"""Verify ASR feasibility validation, memory sampling, and model lifecycle.

Most tests use fakes; the audio decoder test uses generated WAV and MP3 files.
"""

import hashlib
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import av
from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

from test.src import asr_feasibility as feasibility


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (model / name).write_bytes(b"test")
    monkeypatch.setattr(
        feasibility, "MODEL_SHA256", hashlib.sha256(b"test").hexdigest()
    )
    wav, mp3 = tmp_path / "sample.wav", tmp_path / "sample.mp3"
    wav.write_bytes(b"wav")
    mp3.write_bytes(b"mp3")
    return model, [wav, mp3]


def test_inputs_require_pinned_complete_model_and_audio(
    inputs: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, audio = inputs
    feasibility.validate_inputs(model, audio)

    monkeypatch.setattr(feasibility, "MODEL_SHA256", "incorrect")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        feasibility.validate_inputs(model, audio)

    monkeypatch.setattr(
        feasibility, "MODEL_SHA256", hashlib.sha256(b"test").hexdigest()
    )
    (model / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        feasibility.validate_inputs(model, audio)


def test_inputs_require_both_formats_and_nonempty_files(
    inputs: tuple[Path, list[Path]],
) -> None:
    model, audio = inputs
    with pytest.raises(ValueError, match="WAV followed by one MP3"):
        feasibility.validate_inputs(model, list(reversed(audio)))
    audio[0].write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="empty audio"):
        feasibility.validate_inputs(model, audio)


def test_real_audio_decoder_reads_wav_and_mp3(tmp_path: Path) -> None:
    wav_path = tmp_path / "tone.wav"
    mp3_path = tmp_path / "tone.mp3"
    samples = [
        int(10000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(8000)
    ]
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    with av.open(str(wav_path)) as source, av.open(str(mp3_path), "w") as output:
        encoder = output.add_stream("libmp3lame", rate=16000)
        encoder.layout = "mono"
        for frame in source.decode(audio=0):
            for packet in encoder.encode(frame):
                output.mux(packet)
        for packet in encoder.encode(None):
            output.mux(packet)

    for path in (wav_path, mp3_path):
        decoded = decode_audio(str(path), sampling_rate=16000)
        assert 7900 <= len(decoded) <= 8100


def test_gpu_is_required(
    inputs: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "test.src.asr_feasibility.ctranslate2.get_cuda_device_count", lambda: 0
    )
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        feasibility.run_feasibility(*inputs)


def test_memory_sample_reads_gpu_host_and_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "test.src.asr_feasibility.subprocess.run",
        Mock(return_value=SimpleNamespace(stdout="20\n30\n")),
    )
    contents = {
        "/proc/meminfo": "MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n",
        "/sys/fs/cgroup/memory.current": "104857600",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: contents[str(path)])

    assert feasibility.memory_sample() == {
        "gpu_used_mib": 50,
        "host_used_estimate_mib": 512,
        "container_used_mib": 100,
    }


def test_two_jobs_decode_both_formats_and_unload(
    inputs: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    sample: feasibility.MemorySample = {
        "gpu_used_mib": 100,
        "host_used_estimate_mib": 200,
        "container_used_mib": 300,
    }
    unloaded = Mock()
    transcribe = Mock(
        side_effect=lambda *args, **kwargs: (
            iter([SimpleNamespace(text="synthetic speech")]),
            SimpleNamespace(duration=2.0),
        )
    )
    create_model = Mock(
        side_effect=lambda *args, **kwargs: SimpleNamespace(
            model=SimpleNamespace(
                device="cuda", compute_type="int8_float16", unload_model=unloaded
            ),
            transcribe=transcribe,
        )
    )
    decode = Mock(return_value="audio")
    monkeypatch.setattr(
        "test.src.asr_feasibility.ctranslate2.get_cuda_device_count", lambda: 1
    )
    monkeypatch.setattr(feasibility, "memory_sample", lambda: sample)
    monkeypatch.setattr(feasibility, "collect_samples", lambda stop: [sample])
    monkeypatch.setattr(feasibility, "WhisperModel", create_model)
    monkeypatch.setattr(feasibility, "decode_audio", decode)
    monkeypatch.setattr("test.src.asr_feasibility.time.sleep", lambda _: None)

    report = feasibility.run_feasibility(*inputs)

    assert len(report["jobs"]) == 2
    assert all(job["effective_device"] == "cuda" for job in report["jobs"])
    assert all(
        job["effective_compute_type"] == "int8_float16" for job in report["jobs"]
    )
    assert all(
        [r["format"] for r in job["recordings"]] == [".wav", ".mp3"]
        for job in report["jobs"]
    )
    assert report["sampled_peak_memory"] == sample
    assert create_model.call_count == unloaded.call_count == 2
    assert decode.call_count == transcribe.call_count == 4
    assert transcribe.call_args.kwargs["vad_filter"] is True
    assert create_model.call_args.kwargs["device"] == "cuda"
    assert create_model.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize(
    "effective_type,segments", [("float16", ["speech"]), ("int8_float16", [])]
)
def test_invalid_compute_or_empty_output_fails_and_unloads(
    inputs: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
    effective_type: str,
    segments: list[str],
) -> None:
    sample: feasibility.MemorySample = {
        "gpu_used_mib": 100,
        "host_used_estimate_mib": 200,
        "container_used_mib": 300,
    }
    unload = Mock()
    model = SimpleNamespace(
        model=SimpleNamespace(
            device="cuda", compute_type=effective_type, unload_model=unload
        ),
        transcribe=Mock(
            return_value=(
                iter(SimpleNamespace(text=text) for text in segments),
                SimpleNamespace(duration=2.0),
            )
        ),
    )
    monkeypatch.setattr(
        "test.src.asr_feasibility.ctranslate2.get_cuda_device_count", lambda: 1
    )
    monkeypatch.setattr(feasibility, "memory_sample", lambda: sample)
    monkeypatch.setattr(feasibility, "collect_samples", lambda stop: [sample])
    monkeypatch.setattr(feasibility, "WhisperModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(feasibility, "decode_audio", lambda *args, **kwargs: "audio")

    with pytest.raises(RuntimeError, match="Expected|No speech"):
        feasibility.run_feasibility(*inputs)

    unload.assert_called_once()
