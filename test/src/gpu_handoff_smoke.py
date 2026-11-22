"""Exercise sequential GPU stages and cancellable Ollama generation offline."""

import asyncio
import gc
import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import httpx

from app.asr_contract import CUDA_DEVICE, Transcript
from app.asr_service import TranscriptionService
from app.community1_diarizer import Community1Diarizer
from app.diarization_service import DiarizationService
from app.faster_whisper_asr import FasterWhisperTranscriber
from app.inference_gate import InferenceGate
from app.ollama_client import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_DONE_REASON_STOP,
    OllamaClient,
    OllamaGenerationResult,
)
from app.speaker_alignment import align_speakers

ASR_MODEL_DIR = Path("/app/models/faster-whisper-large-v3-turbo")
DIARIZATION_MODEL_DIR = Path("/app/models/pyannote-community-1")
AUDIO_PATH = Path("/audio/two-voices.wav")
CONTEXT_SIZE = 4096
MAX_GENERATED_TOKENS = 32


def set_stage(name: str) -> None:
    print(f"MEETSCRIBE_STAGE {name}", flush=True)


@contextmanager
def inference_lease(gate: InferenceGate) -> Iterator[None]:
    if not gate.try_acquire():
        raise RuntimeError("Another inference stage owns the GPU gate.")
    try:
        yield
    finally:
        gate.release()


def memory_snapshot() -> dict[str, float]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[name] = int(value.split()[0])
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required.issubset(memory):
        raise RuntimeError("Host memory information is incomplete.")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpu_values = [float(line) for line in gpu.stdout.splitlines() if line.strip()]
    if not gpu_values:
        raise RuntimeError("No GPU memory measurements are available.")
    return {
        "gpu_used_mib": sum(gpu_values),
        "host_used_mib": (memory["MemTotal"] - memory["MemAvailable"]) / 1024,
        "swap_used_mib": (memory["SwapTotal"] - memory["SwapFree"]) / 1024,
        "app_container_mib": int(Path("/sys/fs/cgroup/memory.current").read_text())
        / 1024**2,
    }


def run_asr_stage(
    stage: str,
) -> tuple[Transcript, dict[str, object], dict[str, float]]:
    set_stage(stage)
    started = time.monotonic()
    transcriber = FasterWhisperTranscriber(ASR_MODEL_DIR, CUDA_DEVICE)
    try:
        transcript = TranscriptionService(transcriber).transcribe(AUDIO_PATH, "en")
        if not transcript.segments:
            raise RuntimeError("The GPU ASR stage returned no segments.")
        if transcript.provenance.device != CUDA_DEVICE:
            raise RuntimeError("ASR did not run on the requested GPU.")
        inference_memory = memory_snapshot()
        report = {
            "device": transcript.provenance.device,
            "compute_type": transcript.provenance.compute_type,
            "segments": len(transcript.segments),
            "provenance": asdict(transcript.provenance),
            "memory_during_inference": inference_memory,
        }
    finally:
        transcriber.close()
    del transcriber
    gc.collect()
    elapsed = time.monotonic() - started
    post_stage = f"{stage}-post"
    set_stage(post_stage)
    time.sleep(0.3)
    return transcript, {"seconds": round(elapsed, 3), **report}, memory_snapshot()


def run_diarization_stage(
    transcript: Transcript,
) -> tuple[dict[str, object], dict[str, float]]:
    set_stage("diarization")
    started = time.monotonic()
    diarizer = Community1Diarizer(DIARIZATION_MODEL_DIR, CUDA_DEVICE)
    try:
        result = DiarizationService(diarizer).diarize(AUDIO_PATH)
        if not result.turns or not result.exclusive_turns:
            raise RuntimeError("The GPU diarization stage returned no speaker turns.")
        if result.provenance.device != CUDA_DEVICE:
            raise RuntimeError("Diarization did not run on the requested GPU.")
        attributed = align_speakers(transcript, result)
        if len(attributed.attributions) != len(transcript.segments):
            raise RuntimeError("Speaker alignment returned an incomplete result.")
        inference_memory = memory_snapshot()
        report = {
            "device": result.provenance.device,
            "turns": len(result.turns),
            "exclusive_turns": len(result.exclusive_turns),
            "aligned_segments": len(attributed.attributions),
            "provenance": asdict(result.provenance),
            "memory_during_inference": inference_memory,
        }
    finally:
        diarizer.close()
    del diarizer
    gc.collect()
    elapsed = time.monotonic() - started
    set_stage("diarization-post")
    time.sleep(0.3)
    return {"seconds": round(elapsed, 3), **report}, memory_snapshot()


async def loaded_model_names() -> list[str]:
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=5) as client:
        response = await client.get("/api/ps")
        response.raise_for_status()
        payload: object = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise RuntimeError("Ollama returned an invalid loaded-model list.")
    models = payload["models"]
    names: list[str] = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise RuntimeError("Ollama returned an invalid loaded-model entry.")
        names.append(model["name"])
    return names


def generation_report(
    result: OllamaGenerationResult, seconds: float
) -> dict[str, object]:
    if result.done_reason != OLLAMA_DONE_REASON_STOP or not result.text:
        raise RuntimeError("Ollama generation did not complete successfully.")
    return {
        "seconds": round(seconds, 3),
        "done_reason": result.done_reason,
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "response_characters": len(result.text),
    }


async def run_ollama(
    gate: InferenceGate,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, float]]]:
    client = OllamaClient(gate)
    timings: dict[str, dict[str, object]] = {}
    snapshots: dict[str, dict[str, float]] = {}
    try:

        def capture_initial(_chunk: str) -> None:
            if "during_initial" not in snapshots:
                snapshots["during_initial"] = memory_snapshot()

        set_stage("ollama-initial")
        started = time.monotonic()
        first_result = await client.generate(
            "Return a JSON object with the single field ready set to true.",
            num_ctx=CONTEXT_SIZE,
            num_predict=MAX_GENERATED_TOKENS,
            response_format="json",
            on_text=capture_initial,
        )
        timings["initial"] = generation_report(first_result, time.monotonic() - started)
        if "during_initial" not in snapshots:
            raise RuntimeError("Could not sample memory during Ollama generation.")
        if MODEL_NAME in await loaded_model_names():
            raise RuntimeError("Ollama retained the model after generation.")
        set_stage("ollama-initial-post")
        time.sleep(0.3)
        snapshots["after_initial"] = memory_snapshot()

        async def regenerate(
            prompt: str, label: str
        ) -> tuple[OllamaGenerationResult, float]:
            def capture(_chunk: str) -> None:
                snapshot_name = f"during_{label}"
                if snapshot_name not in snapshots:
                    snapshots[snapshot_name] = memory_snapshot()

            started_at = time.monotonic()
            result = await client.generate(
                prompt,
                num_ctx=CONTEXT_SIZE,
                num_predict=MAX_GENERATED_TOKENS,
                response_format="json",
                on_text=capture,
            )
            return result, time.monotonic() - started_at

        set_stage("ollama-concurrent")
        concurrent_results = await asyncio.gather(
            regenerate(
                "Return a JSON object with ready set to true.", "concurrent_first"
            ),
            regenerate(
                "Return a JSON object with revision set to 2.", "concurrent_second"
            ),
        )
        timings["concurrent_first"] = generation_report(*concurrent_results[0])
        timings["concurrent_second"] = generation_report(*concurrent_results[1])
        if not {"during_concurrent_first", "during_concurrent_second"}.issubset(
            snapshots
        ):
            raise RuntimeError("Could not sample memory during concurrent generations.")
        if MODEL_NAME in await loaded_model_names():
            raise RuntimeError("Ollama retained the model after regeneration.")
        set_stage("ollama-concurrent-post")
        time.sleep(0.3)
        snapshots["after_concurrent"] = memory_snapshot()

        def capture_cancellation(_chunk: str) -> None:
            if "during_cancellation" not in snapshots:
                snapshots["during_cancellation"] = memory_snapshot()
            first_text.set()

        set_stage("ollama-cancel")
        first_text = asyncio.Event()
        cancel_started = time.monotonic()
        task = asyncio.create_task(
            client.generate(
                "Write a detailed long report with many sections and paragraphs. "
                "Continue until every section is complete.",
                num_ctx=CONTEXT_SIZE,
                num_predict=4096,
                on_text=capture_cancellation,
            )
        )
        first_text_task = asyncio.create_task(first_text.wait())
        done, _ = await asyncio.wait(
            (task, first_text_task),
            timeout=180,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            first_text_task.cancel()
            task.cancel()
            try:
                await first_text_task
            except asyncio.CancelledError:
                pass
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise RuntimeError(
                "Ollama produced no text before the cancellation timeout."
            )
        if first_text_task not in done:
            first_text_task.cancel()
            try:
                await first_text_task
            except asyncio.CancelledError:
                pass
            await task
            raise RuntimeError("Ollama ended before producing text.")
        await first_text_task
        if task.done():
            await task
            raise RuntimeError(
                "The long generation finished before it could be cancelled."
            )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        else:
            cancelled = False
        if not cancelled:
            raise RuntimeError("The Ollama generation was not cancelled.")
        if "during_cancellation" not in snapshots:
            raise RuntimeError("Could not sample memory during cancelled generation.")
        if MODEL_NAME in await loaded_model_names():
            raise RuntimeError("Ollama retained the model after cancellation.")
        if not gate.try_acquire():
            raise RuntimeError("The GPU gate remained held after cancellation.")
        gate.release()
        timings["cancelled_generation"] = {
            "seconds": round(time.monotonic() - cancel_started, 3),
            "cancelled": True,
            "model_released": True,
        }
        set_stage("ollama-cancel-post")
        time.sleep(0.3)
        snapshots["after_cancellation"] = memory_snapshot()
        return timings, snapshots
    finally:
        await client.aclose()


def main() -> None:
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(f"Missing integration audio: {AUDIO_PATH}")
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    gate = InferenceGate()
    with inference_lease(gate):
        transcript_before, asr_before, asr_before_memory = run_asr_stage("asr-before")
        diarization, diarization_memory = run_diarization_stage(transcript_before)
    del transcript_before
    ollama, ollama_memory = asyncio.run(run_ollama(gate))
    with inference_lease(gate):
        transcript_after, asr_after, asr_after_memory = run_asr_stage("asr-after")
    del transcript_after
    print(
        "MEETSCRIBE_RESULT "
        + json.dumps(
            {
                "asr_before": asr_before,
                "diarization": diarization,
                "ollama": ollama,
                "asr_after": asr_after,
                "post_stage_memory": {
                    "asr_before": asr_before_memory,
                    "diarization": diarization_memory,
                    **ollama_memory,
                    "asr_after": asr_after_memory,
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
