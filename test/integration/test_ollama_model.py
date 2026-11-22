"""Verify the pinned local Ollama model using the app-only offline network.

Requires MEETSCRIBE_OLLAMA_TESTS=1, a provisioned model, Docker, and an NVIDIA GPU.
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "app/model_manifests/summary_model.toml"
SMOKE_PATH = PROJECT_ROOT / "test/src/ollama_smoke.py"
READY_CHECK = (
    "import urllib.request; "
    "urllib.request.urlopen('http://ollama:11434/api/version', timeout=2).read()"
)


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_OLLAMA_TESTS") != "1",
    reason="Set MEETSCRIBE_OLLAMA_TESTS=1 to run the Ollama model test.",
)
def test_pinned_ollama_model_runs_offline_on_gpu() -> None:
    import tomllib

    manifest = tomllib.loads(MANIFEST_PATH.read_text())
    model_name = manifest["model"]
    model_digest = manifest["model_digest"]
    smoke_settings = manifest["smoke"]

    project = f"meetscribe-ollama-{uuid.uuid4().hex[:12]}"
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
    ]
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

        app_id = subprocess.run(
            [*compose, "ps", "-q", "app"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        ollama_id = subprocess.run(
            [*compose, "ps", "-q", "ollama"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        app_container = json.loads(
            subprocess.run(
                ["docker", "inspect", app_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )[0]
        app_networks = app_container["NetworkSettings"]["Networks"]
        assert len(app_networks) == 1
        inference_network = next(iter(app_networks))
        network = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                inference_network,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        network_info = json.loads(network.stdout)[0]
        assert network_info["Internal"] is True
        assert set(network_info["Containers"]) == {app_id, ollama_id}
        ollama_container = json.loads(
            subprocess.run(
                ["docker", "inspect", ollama_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )[0]
        assert not ollama_container["HostConfig"]["PortBindings"]

        smoke = subprocess.run(
            [
                *compose,
                "exec",
                "-T",
                "app",
                "python",
                "-",
                model_name,
                model_digest,
                str(smoke_settings["context_size"]),
                str(smoke_settings["max_generated_tokens"]),
                str(smoke_settings["temperature"]),
                smoke_settings["keep_alive"],
                str(smoke_settings["think"]).lower(),
                smoke_settings["response_format"],
            ],
            cwd=PROJECT_ROOT,
            env=env,
            input=SMOKE_PATH.read_text(),
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        result = json.loads(smoke.stdout)
        assert result["model_digest"] == model_digest.removeprefix("sha256:")
        assert result["quantization"] == "Q4_K_M"
        assert result["context_size"] == smoke_settings["context_size"]
        assert result["think"] is smoke_settings["think"]
        assert result["response_format"] == smoke_settings["response_format"]
        assert result["done_reason"] == "stop"
        assert result["response_nonempty"] is True

        placement = subprocess.run(
            [*compose, "exec", "-T", "ollama", "ollama", "ps"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert "100% GPU" in placement.stdout
        logs = subprocess.run(
            [*compose, "logs", "--no-color", "ollama"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        placement_logs = [
            line.strip()
            for line in logs.stdout.splitlines()
            if any(
                marker in line.lower()
                for marker in ("offloaded", "kv buffer", "cpu buffer", "cuda0 buffer")
            )
        ]

        gpu_memory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_memory = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                ollama_id,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(
            json.dumps(
                {
                    "offline_network": network_info["Internal"],
                    "network_container_count": len(network_info["Containers"]),
                    "model_smoke": result,
                    "model_placement": placement.stdout.strip(),
                    "placement_logs": placement_logs,
                    "gpu_memory_used_free": gpu_memory.stdout.strip(),
                    "ollama_container_memory": container_memory.stdout.strip(),
                },
                indent=2,
            )
        )
    finally:
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
