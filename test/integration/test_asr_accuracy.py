"""Compare offline GPU transcripts of synthetic WAV and MP3 with known text.

Requires MEETSCRIBE_ASR_TESTS=1, a provisioned model, Docker, and an NVIDIA GPU.
Uses the session-scoped ASR image fixture from conftest.py.
"""

import json
import os
import subprocess
import unicodedata
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "test/data"
EXPECTED_TEXT = "This is a test meeting. The project is on schedule."

TRANSCRIBE_SCRIPT = """
import json
from pathlib import Path
from asr_feasibility import COMPUTE_TYPE, validate_inputs
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

model_dir = Path("/model")
audio_paths = [Path("/audio/synthetic-meeting.wav"), Path("/audio/synthetic-meeting.mp3")]
validate_inputs(model_dir, audio_paths)
model = WhisperModel(str(model_dir), device="cuda", compute_type=COMPUTE_TYPE, local_files_only=True)
if model.model.device != "cuda" or model.model.compute_type != COMPUTE_TYPE:
    raise RuntimeError("The requested CUDA compute type is unavailable.")
results = {}
for path in audio_paths:
    audio = decode_audio(str(path), sampling_rate=16000)
    segments, _ = model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
    results[path.suffix] = "".join(segment.text for segment in segments).strip()
print(json.dumps(results))
"""


def normalize_text(text: str) -> str:
    without_punctuation = "".join(
        character
        for character in text.casefold()
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_ASR_TESTS") != "1",
    reason="Set MEETSCRIBE_ASR_TESTS=1 to run the offline GPU integration test.",
)
def test_synthetic_speech_transcription(asr_image: str) -> None:
    model_dir = Path(
        os.environ.get(
            "MEETSCRIBE_MODEL_DIR",
            str(PROJECT_ROOT / "models/faster-whisper-large-v3-turbo"),
        )
    ).resolve(strict=True)
    sample_dir = SAMPLE_DIR.resolve(strict=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--mount",
            f"type=bind,src={PROJECT_ROOT / 'test/src'},dst=/checks,readonly",
            "--env",
            "PYTHONPATH=/checks",
            "--mount",
            f"type=bind,src={model_dir},dst=/model,readonly",
            "--mount",
            f"type=bind,src={sample_dir},dst=/audio,readonly",
            "--entrypoint",
            "python",
            asr_image,
            "-c",
            TRANSCRIBE_SCRIPT,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"GPU container failed:\n{result.stderr}\n{result.stdout}"
    )
    transcripts = json.loads(result.stdout)
    assert set(transcripts) == {".wav", ".mp3"}
    for audio_format, actual in transcripts.items():
        assert normalize_text(actual) == normalize_text(EXPECTED_TEXT), (
            f"{audio_format}: expected {EXPECTED_TEXT!r}, got {actual!r}"
        )
