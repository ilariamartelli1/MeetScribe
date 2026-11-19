"""Run the Community-1 adapter on the synthetic two-voice fixture."""

import json
from dataclasses import asdict
from pathlib import Path

from app.asr_contract import CUDA_DEVICE
from app.community1_diarizer import Community1Diarizer
from app.diarization_service import DiarizationService

MODEL_DIR = Path("/diarization")
AUDIO_PATH = Path("/audio/two-voices.wav")

diarizer = Community1Diarizer(MODEL_DIR, CUDA_DEVICE)
try:
    result = DiarizationService(diarizer).diarize(AUDIO_PATH)
    print(json.dumps(asdict(result)))
finally:
    diarizer.close()
