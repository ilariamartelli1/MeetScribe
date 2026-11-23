"""Run synthetic audio through the local API and report transcript/minutes quality."""

import json
import os
import subprocess
import threading
import time
import wave
from _thread import LockType
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

import httpx

API_BASE_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 0.5
TRANSCRIPTION_TIMEOUT_SECONDS = 900
MINUTES_TIMEOUT_SECONDS = 300
SAMPLE_INTERVAL_SECONDS = 1.0
REPORT_PREFIX = "MEETSCRIBE_MINUTES_QUALITY_RESULT "


class EvaluationCase(TypedDict):
    name: str
    language: str
    audio_path: Path
    reference_path: Path
    metadata_path: Path | None


class ResourceSample(TypedDict):
    stage: str
    elapsed_seconds: float
    gpu_used_mib: float
    host_used_mib: float
    swap_used_mib: float
    app_container_mib: float


MetricName = Literal[
    "gpu_used_mib",
    "host_used_mib",
    "swap_used_mib",
    "app_container_mib",
]


CASES: tuple[EvaluationCase, ...] = (
    {
        "name": "english-checkpoint",
        "language": "en",
        "audio_path": Path("/audio/checkpoint/english/meeting-en.wav"),
        "reference_path": Path("/audio/checkpoint/english/meeting-en-script.txt"),
        "metadata_path": Path(
            "/audio/checkpoint/english/meeting-en-draft-reference.json"
        ),
    },
    {
        "name": "italian-checkpoint",
        "language": "it",
        "audio_path": Path("/audio/checkpoint/italian/meeting-it.wav"),
        "reference_path": Path("/audio/checkpoint/italian/meeting-it-script.txt"),
        "metadata_path": Path(
            "/audio/checkpoint/italian/meeting-it-draft-reference.json"
        ),
    },
    {
        "name": "known-asr-error",
        "language": "en",
        "audio_path": Path("/audio/smoke/two-voices.wav"),
        "reference_path": Path("/audio/smoke/two-voices-script.txt"),
        "metadata_path": None,
    },
)
CASE_SELECTION_ENV = "MEETSCRIBE_MINUTES_QUALITY_CASES"


def _selected_cases() -> tuple[EvaluationCase, ...]:
    selection = os.environ.get(CASE_SELECTION_ENV)
    if selection is None:
        return CASES

    selected_names = {name.strip() for name in selection.split(",") if name.strip()}
    if not selected_names:
        raise ValueError(f"{CASE_SELECTION_ENV} must name at least one case.")
    available_names = {case["name"] for case in CASES}
    unknown_names = selected_names - available_names
    if unknown_names:
        raise ValueError(
            f"Unknown {CASE_SELECTION_ENV} case(s): {', '.join(sorted(unknown_names))}"
        )
    return tuple(case for case in CASES if case["name"] in selected_names)


def _set_stage(stage_state: dict[str, str], stage_lock: LockType, name: str) -> None:
    with stage_lock:
        stage_state["name"] = name


def _resource_sample(stage: str, elapsed_seconds: float) -> ResourceSample:
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

    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key] = int(value.split()[0])
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required.issubset(memory):
        raise RuntimeError("Host memory information is incomplete.")

    app_memory = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    return {
        "stage": stage,
        "elapsed_seconds": elapsed_seconds,
        "gpu_used_mib": sum(gpu_values),
        "host_used_mib": (memory["MemTotal"] - memory["MemAvailable"]) / 1024,
        "swap_used_mib": (memory["SwapTotal"] - memory["SwapFree"]) / 1024,
        "app_container_mib": app_memory / 1024**2,
    }


def _wait_for_job(
    client: httpx.Client, url: str, expected_state: str, timeout_seconds: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(url)
        response.raise_for_status()
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("The API returned an invalid job status.")
        state = payload.get("state")
        if state == expected_state:
            return payload
        if state in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Job did not reach {expected_state}: {json.dumps(payload)}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Job did not reach {expected_state} before the timeout.")


def _audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Audio has an invalid frame rate: {path.name}")
        return audio.getnframes() / frame_rate


def _reference_status(case: EvaluationCase) -> str:
    metadata_path = case["metadata_path"]
    if metadata_path is None:
        return (
            "Prior exact-text diagnostic reported substitutions for Friday and "
            "Elena; assess this run against the source script."
        )
    metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Reference metadata is invalid: {metadata_path.name}")
    status = metadata.get("status")
    if not isinstance(status, str):
        raise ValueError(f"Reference metadata is invalid: {metadata_path.name}")
    return status


def _run_case(
    client: httpx.Client,
    case: EvaluationCase,
    set_stage: Callable[[str], None],
    started_at: float,
) -> dict[str, object]:
    name = case["name"]
    language = case["language"]
    audio_path = case["audio_path"]
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing synthetic test audio: {audio_path}")
    reference_path = case["reference_path"]
    reference_text = reference_path.read_text(encoding="utf-8")
    set_stage(f"{name}-transcription")

    origin = {"origin": API_BASE_URL}
    transcription_started = time.monotonic()
    with audio_path.open("rb") as audio:
        uploaded = client.post(
            "/api/v1/jobs",
            headers=origin,
            data={"language": language},
            files={"file": (audio_path.name, audio, "audio/wav")},
        )
    if uploaded.status_code != 202:
        raise RuntimeError(f"Audio upload failed: {uploaded.text}")
    job_payload: object = uploaded.json()
    if not isinstance(job_payload, dict) or not isinstance(
        job_payload.get("job_id"), str
    ):
        raise RuntimeError("The upload API returned an invalid job ID.")
    job_id = job_payload["job_id"]
    job_url = f"/api/v1/jobs/{job_id}"
    job_status = _wait_for_job(
        client, job_url, "succeeded", TRANSCRIPTION_TIMEOUT_SECONDS
    )
    if "result" in job_status:
        raise RuntimeError("Normal job status unexpectedly exposed its result.")
    transcription_seconds = time.monotonic() - transcription_started
    transcript_response = client.get(f"{job_url}/transcript.txt")
    transcript_response.raise_for_status()

    set_stage(f"{name}-minutes")
    minutes_started = time.monotonic()
    started = client.post(f"{job_url}/minutes", headers=origin)
    if started.status_code != 202:
        raise RuntimeError(f"Minutes generation did not start: {started.text}")
    minutes_status = _wait_for_job(
        client,
        f"{job_url}/minutes",
        "succeeded",
        MINUTES_TIMEOUT_SECONDS,
    )
    if minutes_status.get("regenerating") is not False:
        raise RuntimeError("Unexpected minutes regeneration state.")
    extraction_metrics = minutes_status.get("extraction_metrics")
    if not isinstance(extraction_metrics, dict):
        raise RuntimeError("Minutes extraction metrics were not returned.")
    export_response = client.get(f"{job_url}/minutes.json")
    export_response.raise_for_status()
    export: object = export_response.json()
    if not isinstance(export, dict):
        raise RuntimeError("The minutes export is not a JSON object.")
    minutes = export.get("minutes")
    segments = export.get("segments")
    if not isinstance(minutes, dict) or not isinstance(segments, list):
        raise RuntimeError("The minutes export is missing its source or summary.")

    return {
        "case": name,
        "language": language,
        "audio_bytes": audio_path.stat().st_size,
        "audio_duration_seconds": round(_audio_duration_seconds(audio_path), 3),
        "transcription_seconds": round(transcription_seconds, 3),
        "minutes_generation_seconds": round(time.monotonic() - minutes_started, 3),
        "reference_status": _reference_status(case),
        "reference_script": reference_text,
        "transcript_download": transcript_response.text,
        "segments": segments,
        "minutes": minutes,
        "review_status": export.get("review_status"),
        "transcript_revision": export.get("transcript_revision"),
        "minutes_source_transcript_revision": export.get(
            "minutes_source_transcript_revision"
        ),
        "stale": export.get("stale"),
        "extraction_metrics": extraction_metrics,
        "after_minutes_resources": _resource_sample(
            f"{name}-minutes-complete", time.monotonic() - started_at
        ),
    }


def _peak_by_stage(samples: list[ResourceSample]) -> dict[str, dict[str, float]]:
    metrics: tuple[MetricName, ...] = (
        "gpu_used_mib",
        "host_used_mib",
        "swap_used_mib",
        "app_container_mib",
    )
    peaks: dict[str, dict[str, float]] = {}
    for sample in samples:
        stage = sample["stage"]
        current = peaks.get(stage)
        if current is None:
            peaks[stage] = {name: sample[name] for name in metrics}
        else:
            peaks[stage] = {name: max(current[name], sample[name]) for name in metrics}
    return peaks


def main() -> None:
    started_at = time.monotonic()
    selected_cases = _selected_cases()
    samples: list[ResourceSample] = []
    sample_errors: list[Exception] = []
    stage_state = {"name": "baseline"}
    stage_lock: LockType = threading.Lock()
    stop_sampler = threading.Event()

    def set_stage(name: str) -> None:
        _set_stage(stage_state, stage_lock, name)

    def sample_resources() -> None:
        while not stop_sampler.is_set():
            with stage_lock:
                stage = stage_state["name"]
            try:
                samples.append(_resource_sample(stage, time.monotonic() - started_at))
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                sample_errors.append(exc)
                stop_sampler.set()
                return
            stop_sampler.wait(SAMPLE_INTERVAL_SECONDS)

    samples.append(_resource_sample("baseline", 0))
    sampler = threading.Thread(target=sample_resources, daemon=True)
    sampler.start()
    try:
        with httpx.Client(
            base_url=API_BASE_URL,
            timeout=httpx.Timeout(30),
        ) as client:
            health = client.get("/api/v1/health")
            health.raise_for_status()
            evaluations = [
                _run_case(client, case, set_stage, started_at)
                for case in selected_cases
            ]
            set_stage("complete")
    finally:
        stop_sampler.set()
        sampler.join(timeout=10)

    if sample_errors:
        raise RuntimeError("Resource sampling failed.") from sample_errors[0]
    if sampler.is_alive():
        raise RuntimeError("Resource sampler did not stop.")
    print(
        REPORT_PREFIX
        + json.dumps(
            {
                "evaluations": evaluations,
                "selected_cases": [case["name"] for case in selected_cases],
                "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
                "sampled_peak_by_stage": _peak_by_stage(samples),
                "resource_samples": samples,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
