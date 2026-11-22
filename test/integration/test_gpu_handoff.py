"""Measure a real ASR -> diarization -> Ollama -> ASR GPU handoff."""

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal, TextIO, TypedDict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = PROJECT_ROOT / "test/src/gpu_handoff_smoke.py"
SAMPLE_INTERVAL_SECONDS = 0.5
PLACEMENT_SAMPLE_INTERVAL_SECONDS = 2
TEST_TIMEOUT_SECONDS = 900
READY_CHECK = (
    "import urllib.request; "
    "urllib.request.urlopen('http://ollama:11434/api/version', timeout=2).read()"
)


class ResourceSample(TypedDict):
    stage: str
    elapsed_seconds: float
    gpu_used_mib: float
    host_used_mib: float
    swap_used_mib: float
    app_container_mib: float
    ollama_container_mib: float


class PlacementSample(TypedDict):
    stage: str
    elapsed_seconds: float
    ollama_ps: str


MetricName = Literal[
    "gpu_used_mib",
    "host_used_mib",
    "swap_used_mib",
    "app_container_mib",
    "ollama_container_mib",
]


def _container_memory_path(container_id: str) -> Path:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Pid}}",
            container_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    pid = result.stdout.strip()
    if not pid.isascii() or not pid.isdigit() or int(pid) < 1:
        raise ValueError(f"Docker returned an invalid container PID: {pid!r}")
    cgroups = (Path("/proc") / pid / "cgroup").read_text().splitlines()
    for line in cgroups:
        hierarchy, separator, group = line.partition("::")
        if hierarchy != "0" or not separator:
            continue
        relative = PurePosixPath(group.lstrip("/"))
        if any(part in {".", ".."} for part in relative.parts):
            raise ValueError("Docker returned an invalid cgroup path.")
        memory_path = Path("/sys/fs/cgroup").joinpath(*relative.parts, "memory.current")
        if memory_path.is_file():
            return memory_path
        break
    raise RuntimeError("Could not locate the container's cgroup memory counter.")


def _container_memory_mib(memory_path: Path) -> float:
    value = int(memory_path.read_text().strip())
    if value < 0:
        raise ValueError(f"Invalid cgroup memory usage: {value}")
    return value / 1024**2


def _resource_sample(
    stage: str,
    elapsed_seconds: float,
    app_memory_path: Path,
    ollama_memory_path: Path,
) -> ResourceSample:
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

    sample: ResourceSample = {
        "stage": stage,
        "elapsed_seconds": elapsed_seconds,
        "gpu_used_mib": sum(gpu_values),
        "host_used_mib": (memory["MemTotal"] - memory["MemAvailable"]) / 1024,
        "swap_used_mib": (memory["SwapTotal"] - memory["SwapFree"]) / 1024,
        "app_container_mib": _container_memory_mib(app_memory_path),
        "ollama_container_mib": _container_memory_mib(ollama_memory_path),
    }
    return sample


def _container_id(compose: list[str], service: str) -> str:
    result = subprocess.run(
        [*compose, "ps", "-q", service],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError(f"Compose did not start the {service} container.")
    return container_id


def _enqueue_output(stream: TextIO, output: queue.Queue[str | None]) -> None:
    for line in stream:
        output.put(line.rstrip("\n"))
    output.put(None)


def _object_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"The GPU handoff test returned invalid {description}.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeError(f"The GPU handoff test returned invalid {description}.")
        result[key] = item
    return result


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_GPU_HANDOFF_TESTS") != "1",
    reason="Set MEETSCRIBE_GPU_HANDOFF_TESTS=1 to run the real GPU handoff test.",
)
def test_real_gpu_handoff_serializes_and_releases_all_models() -> None:
    project = f"meetscribe-gpu-handoff-{uuid.uuid4().hex[:12]}"
    env = os.environ.copy()
    compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--profile",
        "summary",
        "-f",
        str(PROJECT_ROOT / "compose.yaml"),
        "-f",
        str(PROJECT_ROOT / "compose.ollama-offline.yaml"),
        "-f",
        str(PROJECT_ROOT / "compose.gpu-handoff.yaml"),
    ]
    samples: list[ResourceSample] = []
    placement_samples: list[PlacementSample] = []
    sample_errors: list[Exception] = []
    stage_state = {"name": "baseline"}
    stage_lock = threading.Lock()
    stop_sampler = threading.Event()
    sampler_thread: threading.Thread | None = None
    placement_thread: threading.Thread | None = None
    process: subprocess.Popen[str] | None = None
    output_thread: threading.Thread | None = None
    started_at = time.monotonic()
    try:
        subprocess.run(
            [*compose, "up", "--build", "-d", "app", "ollama"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        deadline = time.monotonic() + 60
        last_error = ""
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [*compose, "exec", "-T", "app", "python", "-c", READY_CHECK],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if ready.returncode == 0:
                break
            last_error = ready.stderr
            time.sleep(0.25)
        else:
            pytest.fail(f"Ollama did not become reachable from the app: {last_error}")

        started_at = time.monotonic()
        app_id = _container_id(compose, "app")
        ollama_id = _container_id(compose, "ollama")
        app_memory_path = _container_memory_path(app_id)
        ollama_memory_path = _container_memory_path(ollama_id)
        samples.append(
            _resource_sample("baseline", 0, app_memory_path, ollama_memory_path)
        )

        def sample_resources() -> None:
            while not stop_sampler.is_set():
                with stage_lock:
                    stage = stage_state["name"]
                try:
                    samples.append(
                        _resource_sample(
                            stage,
                            time.monotonic() - started_at,
                            app_memory_path,
                            ollama_memory_path,
                        )
                    )
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

        def sample_ollama_placement() -> None:
            while not stop_sampler.wait(PLACEMENT_SAMPLE_INTERVAL_SECONDS):
                with stage_lock:
                    stage = stage_state["name"]
                if not stage.startswith("ollama-") or stage.endswith("-post"):
                    continue
                try:
                    placement = subprocess.run(
                        ["docker", "exec", ollama_id, "ollama", "ps"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    placement_samples.append(
                        {
                            "stage": stage,
                            "elapsed_seconds": time.monotonic() - started_at,
                            "ollama_ps": placement.stdout.strip(),
                        }
                    )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as exc:
                    sample_errors.append(exc)
                    stop_sampler.set()
                    return

        sampler_thread = threading.Thread(target=sample_resources, daemon=True)
        sampler_thread.start()
        placement_thread = threading.Thread(target=sample_ollama_placement, daemon=True)
        placement_thread.start()
        output: queue.Queue[str | None] = queue.Queue()
        process = subprocess.Popen(
            [*compose, "exec", "-T", "app", "python", "-"],
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Could not open the GPU handoff test streams.")
        process.stdin.write(SMOKE_PATH.read_text(encoding="utf-8"))
        process.stdin.close()
        output_thread = threading.Thread(
            target=_enqueue_output, args=(process.stdout, output), daemon=True
        )
        output_thread.start()

        output_lines: list[str] = []
        report: dict[str, object] | None = None
        deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
        while True:
            try:
                line = output.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("The real GPU handoff test timed out.")
                continue
            if line is None:
                break
            output_lines.append(line)
            if line.startswith("MEETSCRIBE_STAGE "):
                with stage_lock:
                    stage_state["name"] = line.removeprefix("MEETSCRIBE_STAGE ")
            elif line.startswith("MEETSCRIBE_RESULT "):
                decoded: object = json.loads(line.removeprefix("MEETSCRIBE_RESULT "))
                report = _object_mapping(decoded, "JSON")
        return_code = process.wait(timeout=10)
        output_thread.join(timeout=10)
        stop_sampler.set()
        sampler_thread.join(timeout=10)
        placement_thread.join(timeout=10)
        if sample_errors:
            raise RuntimeError("Resource sampling failed.") from sample_errors[0]
        assert return_code == 0, "GPU handoff script failed:\n" + "\n".join(
            output_lines
        )
        assert report is not None, "GPU handoff script returned no report."
        asr_before = _object_mapping(report.get("asr_before"), "ASR result")
        asr_after = _object_mapping(report.get("asr_after"), "ASR result")
        diarization = _object_mapping(report.get("diarization"), "diarization result")
        ollama = _object_mapping(report.get("ollama"), "Ollama result")
        cancelled = _object_mapping(
            ollama.get("cancelled_generation"), "cancellation result"
        )
        assert asr_before.get("device") == "cuda"
        assert asr_after.get("device") == "cuda"
        assert diarization.get("device") == "cuda"
        turns = diarization.get("turns")
        assert isinstance(turns, int) and turns > 0
        assert cancelled.get("cancelled") is True
        assert cancelled.get("model_released") is True
        placements = [sample["ollama_ps"] for sample in placement_samples]
        assert any("100% GPU" in placement for placement in placements), placements

        peak_by_stage: dict[str, dict[str, float]] = {}
        metric_names: tuple[MetricName, ...] = (
            "gpu_used_mib",
            "host_used_mib",
            "swap_used_mib",
            "app_container_mib",
            "ollama_container_mib",
        )
        for sample in samples:
            stage = sample["stage"]
            values: dict[str, float] = {}
            for name in metric_names:
                values[name] = sample[name]
            current = peak_by_stage.get(stage)
            if current is None:
                peak_by_stage[stage] = values
            else:
                peak_by_stage[stage] = {
                    name: max(current[name], values[name]) for name in metric_names
                }
        max_sample_gap_by_stage: dict[str, float] = {}
        previous_sample_by_stage: dict[str, float] = {}
        for sample in samples:
            stage = sample["stage"]
            previous = previous_sample_by_stage.get(stage)
            if previous is not None:
                max_sample_gap_by_stage[stage] = max(
                    max_sample_gap_by_stage.get(stage, 0),
                    sample["elapsed_seconds"] - previous,
                )
            previous_sample_by_stage[stage] = sample["elapsed_seconds"]
        print(
            json.dumps(
                {
                    "result": report,
                    "target_sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
                    "max_sample_gap_by_stage_seconds": max_sample_gap_by_stage,
                    "sampled_peak_by_stage": peak_by_stage,
                    "ollama_placement_samples": placement_samples,
                    "resource_samples": samples,
                },
                indent=2,
            )
        )
    finally:
        stop_sampler.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=10)
        if placement_thread is not None:
            placement_thread.join(timeout=10)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if output_thread is not None:
            output_thread.join(timeout=10)
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
