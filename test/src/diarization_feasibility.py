"""Verify offline CPU diarization after GPU ASR in the application image."""

import gc
import importlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Final

from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

from app.asr_service import TranscriptionService
from app.faster_whisper_asr import FasterWhisperTranscriber
from diarization_model import verify_model  # type: ignore[import-not-found]

ASR_MODEL_DIR = Path("/model")
DIARIZATION_MODEL_DIR = Path("/diarization")
AUDIO_DIR = Path("/audio")
AUDIO_FILENAME_PREFIX = "synthetic-meeting"
AUDIO_FORMATS = ("wav", "mp3")
RUNS_PER_FORMAT = 2
ASR_DEVICE: Final = "cuda"
ASR_LANGUAGE: Final = "en"
DIARIZATION_SAMPLE_RATE_HZ = 16_000
TORCH_THREADS = 2
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.2
POST_STAGE_SAMPLE_WAIT_SECONDS = 0.3
GPU_QUERY_TIMEOUT_SECONDS = 10
TIMING_DECIMAL_PLACES = 2
KIB_PER_MIB = 1024
BYTES_PER_MIB = 1024**2

torch = importlib.import_module("torch")
Pipeline = importlib.import_module("pyannote.audio").Pipeline
task = importlib.import_module("pyannote.audio.core.task")
AudioDecoder = importlib.import_module("torchcodec.decoders").AudioDecoder


def memory() -> dict[str, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    gpu = [float(line) for line in result.stdout.splitlines()]
    if not gpu:
        raise RuntimeError("No GPU memory measurements available.")
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            info[key] = int(value.split()[0])
    return {
        "gpu_mib": sum(gpu),
        "host_used_mib": (info["MemTotal"] - info["MemAvailable"]) / KIB_PER_MIB,
        "swap_used_mib": (info["SwapTotal"] - info["SwapFree"]) / KIB_PER_MIB,
        "container_mib": int(Path("/sys/fs/cgroup/memory.current").read_text())
        / BYTES_PER_MIB,
    }


def main() -> None:
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("PYANNOTE_METRICS_ENABLED") != "0"
    ):
        raise RuntimeError("Offline and disabled telemetry settings are required.")
    verify_model(DIARIZATION_MODEL_DIR)
    if torch.version.cuda is not None:
        raise RuntimeError("Expected CPU-only PyTorch.")
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_THREADS)
    samples: list[dict[str, Any]] = []
    sample_errors: list[Exception] = []
    stage = "baseline"
    stop = threading.Event()

    def sampler() -> None:
        try:
            while not stop.is_set():
                samples.append({"stage": stage, **memory()})
                stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)
        except Exception as exc:
            sample_errors.append(exc)

    thread = threading.Thread(target=sampler)
    thread.start()
    jobs = []
    try:
        for repeat in range(RUNS_PER_FORMAT):
            for extension in AUDIO_FORMATS:
                audio = AUDIO_DIR / f"{AUDIO_FILENAME_PREFIX}.{extension}"
                stage = f"{repeat}-{extension}-codec"
                decoded = AudioDecoder(str(audio)).get_all_samples()
                if decoded.data.numel() == 0:
                    raise RuntimeError(f"TorchCodec decoded no samples from {audio}")
                del decoded
                stage = f"{repeat}-{extension}-asr"
                started = time.monotonic()
                transcriber = FasterWhisperTranscriber(ASR_MODEL_DIR, ASR_DEVICE)
                try:
                    transcript = TranscriptionService(transcriber).transcribe(
                        audio, ASR_LANGUAGE
                    )
                    if not transcript.segments:
                        raise RuntimeError(f"No ASR segments for {audio}")
                finally:
                    transcriber.close()
                del transcriber, transcript
                gc.collect()
                asr_seconds = time.monotonic() - started
                stage = f"{repeat}-{extension}-after-asr"
                time.sleep(POST_STAGE_SAMPLE_WAIT_SECONDS)
                stage = f"{repeat}-{extension}-diarization"
                started = time.monotonic()
                with torch.serialization.safe_globals(
                    [task.Specifications, task.Problem, task.Resolution]
                ):
                    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_DIR)
                if pipeline is None:
                    raise RuntimeError("Failed to load local diarization pipeline.")
                waveform = torch.from_numpy(
                    decode_audio(str(audio), sampling_rate=DIARIZATION_SAMPLE_RATE_HZ)
                ).unsqueeze(0)
                output = pipeline(
                    {"waveform": waveform, "sample_rate": DIARIZATION_SAMPLE_RATE_HZ}
                )
                if (
                    len(output.speaker_diarization) == 0
                    or len(output.exclusive_speaker_diarization) == 0
                ):
                    raise RuntimeError(f"No speaker turns for {audio}")
                turns = len(output.speaker_diarization)
                exclusive_turns = len(output.exclusive_speaker_diarization)
                del output, waveform, pipeline
                gc.collect()
                diarization_seconds = time.monotonic() - started
                stage = f"{repeat}-{extension}-after-diarization"
                time.sleep(POST_STAGE_SAMPLE_WAIT_SECONDS)
                jobs.append(
                    {
                        "format": extension,
                        "asr_seconds": round(asr_seconds, TIMING_DECIMAL_PLACES),
                        "diarization_seconds": round(
                            diarization_seconds, TIMING_DECIMAL_PLACES
                        ),
                        "turns": turns,
                        "exclusive_turns": exclusive_turns,
                        "after": memory(),
                    }
                )
    finally:
        stop.set()
        thread.join()
    if sample_errors:
        raise RuntimeError("Resource sampling failed.") from sample_errors[0]
    if not samples:
        raise RuntimeError("No memory samples collected.")
    stages = {}
    for sample in samples:
        label = sample["stage"]
        metrics = {k: v for k, v in sample.items() if k != "stage"}
        if label not in stages:
            stages[label] = metrics
        else:
            stages[label] = {k: max(stages[label][k], v) for k, v in metrics.items()}
    print(
        json.dumps(
            {"jobs": jobs, "peak_by_stage": stages, "threads": torch.get_num_threads()}
        )
    )


if __name__ == "__main__":
    main()
