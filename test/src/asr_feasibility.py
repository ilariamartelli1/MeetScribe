import argparse
import gc
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import ctranslate2  # type: ignore[import-untyped]
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from faster_whisper.audio import decode_audio  # type: ignore[import-untyped]

MODEL_SHA256 = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
COMPUTE_TYPE = "int8_float16"
SAMPLE_INTERVAL_SECONDS = 0.2


class MemorySample(TypedDict):
    gpu_used_mib: float
    host_used_estimate_mib: float
    container_used_mib: float


class RecordingResult(TypedDict):
    format: str
    decode_seconds: float
    inference_seconds: float
    duration_seconds: float
    segments: int


class JobResult(TypedDict):
    effective_device: str
    effective_compute_type: str
    load_seconds: float
    recordings: list[RecordingResult]
    unload_seconds: float
    memory_after_unload: MemorySample


class FeasibilityReport(TypedDict):
    requested_device: str
    requested_compute_type: str
    sampling_interval_seconds: float
    baseline_memory: MemorySample
    sampled_peak_memory: MemorySample
    jobs: list[JobResult]


def validate_inputs(model_dir: Path, audio_paths: list[Path]) -> None:
    for filename in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing provisioned model file: {model_dir / filename}"
            )
    with (model_dir / "model.bin").open("rb") as model_file:
        actual = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual != MODEL_SHA256:
        raise ValueError("Model weights do not match the pinned SHA-256.")
    if len(audio_paths) != 2 or [path.suffix.lower() for path in audio_paths] != [
        ".wav",
        ".mp3",
    ]:
        raise ValueError("Provide one WAV followed by one MP3 sample.")
    for path in audio_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty audio sample: {path}")


def memory_sample() -> MemorySample:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpu_lines = gpu.stdout.strip().splitlines()
    if not gpu_lines:
        raise RuntimeError("NVIDIA GPU memory information is unavailable.")
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable"}:
            memory[name] = int(value.split()[0])
    return {
        "gpu_used_mib": sum(float(line) for line in gpu_lines),
        "host_used_estimate_mib": (memory["MemTotal"] - memory["MemAvailable"]) / 1024,
        "container_used_mib": int(Path("/sys/fs/cgroup/memory.current").read_text())
        / 1024**2,
    }


def collect_samples(stop: threading.Event) -> list[MemorySample]:
    samples = [memory_sample()]
    while not stop.wait(SAMPLE_INTERVAL_SECONDS):
        samples.append(memory_sample())
    return samples


def run_feasibility(model_dir: Path, audio_paths: list[Path]) -> FeasibilityReport:
    validate_inputs(model_dir, audio_paths)
    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("A CUDA GPU must be available; CPU fallback is disabled.")

    baseline = memory_sample()
    stop = threading.Event()
    jobs: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        samples_future = executor.submit(collect_samples, stop)
        try:
            for _ in range(2):
                start = time.monotonic()
                model = WhisperModel(
                    str(model_dir),
                    device="cuda",
                    compute_type=COMPUTE_TYPE,
                    local_files_only=True,
                )
                load_seconds = time.monotonic() - start
                try:
                    effective_device = model.model.device
                    if effective_device != "cuda":
                        raise RuntimeError(
                            f"Expected CUDA device, got {effective_device}."
                        )
                    effective_compute_type = model.model.compute_type
                    if effective_compute_type != COMPUTE_TYPE:
                        raise RuntimeError(
                            f"Expected {COMPUTE_TYPE}, got {effective_compute_type}."
                        )
                    recordings: list[RecordingResult] = []
                    for audio_path in audio_paths:
                        start = time.monotonic()
                        audio = decode_audio(str(audio_path), sampling_rate=16000)
                        decode_seconds = time.monotonic() - start
                        start = time.monotonic()
                        segments, info = model.transcribe(
                            audio, language="en", beam_size=5, vad_filter=True
                        )
                        result = list(segments)
                        inference_seconds = time.monotonic() - start
                        if not result or not any(
                            segment.text.strip() for segment in result
                        ):
                            raise RuntimeError(
                                f"No speech transcribed from {audio_path.name}."
                            )
                        recordings.append(
                            {
                                "format": audio_path.suffix.lower(),
                                "decode_seconds": round(decode_seconds, 3),
                                "inference_seconds": round(inference_seconds, 3),
                                "duration_seconds": round(info.duration, 3),
                                "segments": len(result),
                            }
                        )
                finally:
                    start = time.monotonic()
                    model.model.unload_model()
                    del model
                    gc.collect()
                    unload_seconds = time.monotonic() - start
                time.sleep(SAMPLE_INTERVAL_SECONDS)
                jobs.append(
                    {
                        "effective_device": effective_device,
                        "effective_compute_type": effective_compute_type,
                        "load_seconds": round(load_seconds, 3),
                        "recordings": recordings,
                        "unload_seconds": round(unload_seconds, 3),
                        "memory_after_unload": memory_sample(),
                    }
                )
        finally:
            stop.set()
        samples = samples_future.result()

    return {
        "requested_device": "cuda",
        "requested_compute_type": COMPUTE_TYPE,
        "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "baseline_memory": baseline,
        "sampled_peak_memory": {
            "gpu_used_mib": max(sample["gpu_used_mib"] for sample in samples),
            "host_used_estimate_mib": max(
                sample["host_used_estimate_mib"] for sample in samples
            ),
            "container_used_mib": max(
                sample["container_used_mib"] for sample in samples
            ),
        },
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline GPU ASR feasibility checks."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--mp3", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_feasibility(args.model_dir, [args.wav, args.mp3])
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"ASR feasibility check failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
