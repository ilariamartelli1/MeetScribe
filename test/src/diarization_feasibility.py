"""Verify offline ASR and diarization on an explicitly selected device."""

import argparse
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

from app.asr_contract import CPU_DEVICE, CUDA_DEVICE, Device
from app.asr_service import TranscriptionService
from app.community1_diarizer import MODEL_MANIFEST
from app.faster_whisper_asr import FasterWhisperTranscriber
from app.model_integrity import verify_model

ASR_MODEL_DIR = Path("/model")
DIARIZATION_MODEL_DIR = Path("/diarization")
AUDIO_DIR = Path("/audio")
AUDIO_FILENAME_PREFIX = "synthetic-meeting"
AUDIO_FORMATS = ("wav", "mp3")
RUNS_PER_FORMAT = 2
DEFAULT_DEVICE: Device = CUDA_DEVICE
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


def memory(device: Device) -> dict[str, float]:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            info[key] = int(value.split()[0])
    measurements = {
        "host_used_mib": (info["MemTotal"] - info["MemAvailable"]) / KIB_PER_MIB,
        "swap_used_mib": (info["SwapTotal"] - info["SwapFree"]) / KIB_PER_MIB,
        "container_mib": int(Path("/sys/fs/cgroup/memory.current").read_text())
        / BYTES_PER_MIB,
    }
    if device == CUDA_DEVICE:
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
        measurements["gpu_mib"] = sum(gpu)
    return measurements


def main(device: Device, formats: tuple[str, ...], runs_per_format: int) -> None:
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("PYANNOTE_METRICS_ENABLED") != "0"
    ):
        raise RuntimeError("Offline and disabled telemetry settings are required.")
    if device == CUDA_DEVICE and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable for diarization; CPU fallback is disabled. "
            "Select --device cpu explicitly to use CPU."
        )
    verify_model(DIARIZATION_MODEL_DIR, MODEL_MANIFEST)
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_THREADS)
    samples: list[dict[str, Any]] = []
    sample_errors: list[Exception] = []
    stage = "baseline"
    stop = threading.Event()

    def sampler() -> None:
        try:
            while not stop.is_set():
                samples.append({"stage": stage, **memory(device)})
                stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)
        except Exception as exc:
            sample_errors.append(exc)

    thread = threading.Thread(target=sampler)
    thread.start()
    jobs = []
    try:
        for repeat in range(runs_per_format):
            for extension in formats:
                audio = AUDIO_DIR / f"{AUDIO_FILENAME_PREFIX}.{extension}"
                stage = f"{repeat}-{extension}-codec"
                decoded = AudioDecoder(str(audio)).get_all_samples()
                if decoded.data.numel() == 0:
                    raise RuntimeError(f"TorchCodec decoded no samples from {audio}")
                del decoded
                stage = f"{repeat}-{extension}-asr"
                started = time.monotonic()
                transcriber = FasterWhisperTranscriber(ASR_MODEL_DIR, device)
                try:
                    transcript = TranscriptionService(transcriber).transcribe(
                        audio, ASR_LANGUAGE
                    )
                    if not transcript.segments:
                        raise RuntimeError(f"No ASR segments for {audio}")
                    if transcript.provenance.device != device:
                        raise RuntimeError("ASR did not use the selected device.")
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
                pipeline.to(torch.device(device))
                if (
                    pipeline.device.type != device
                    or pipeline._segmentation.model.device.type != device
                    or pipeline._embedding.device.type != device
                ):
                    raise RuntimeError("Diarization did not use the selected device.")
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
                if device == CUDA_DEVICE:
                    torch.cuda.synchronize()
                del output, waveform, pipeline
                gc.collect()
                if device == CUDA_DEVICE:
                    torch.cuda.empty_cache()
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
                        "after": memory(device),
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
            {
                "device": device,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "jobs": jobs,
                "peak_by_stage": stages,
                "threads": torch.get_num_threads(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=(CUDA_DEVICE, CPU_DEVICE), default=DEFAULT_DEVICE
    )
    parser.add_argument("--format", choices=AUDIO_FORMATS)
    parser.add_argument("--runs-per-format", type=int, default=RUNS_PER_FORMAT)
    args = parser.parse_args()
    if args.runs_per_format < 1:
        parser.error("--runs-per-format must be at least 1")
    main(
        args.device,
        (args.format,) if args.format else AUDIO_FORMATS,
        args.runs_per_format,
    )
