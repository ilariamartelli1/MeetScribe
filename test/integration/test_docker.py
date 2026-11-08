"""Verify the CPU container health endpoint and loopback-only port binding.

Requires MEETSCRIBE_DOCKER_TESTS=1 and Docker.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest


@pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_DOCKER_TESTS") != "1",
    reason="Set MEETSCRIBE_DOCKER_TESTS=1 to run the Docker smoke test.",
)
def test_cpu_container_is_healthy_and_bound_to_loopback() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    env = os.environ | {"MEETSCRIBE_PORT": str(port)}
    compose = [
        "docker",
        "compose",
        "--project-name",
        f"meetscribe-smoke-{uuid.uuid4().hex[:12]}",
        "-f",
        "compose.yaml",
        "-f",
        "compose.cpu.yaml",
    ]
    try:
        subprocess.run(
            [*compose, "up", "--build", "-d"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        response_body = None
        last_error: Exception | None = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/health", timeout=2
                ) as response:
                    response_body = json.load(response)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.2)
        assert response_body == {"status": "ok"}, f"Last connection error: {last_error}"

        binding = subprocess.run(
            [*compose, "port", "app", "8000"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert binding.stdout.strip() == f"127.0.0.1:{port}"
    finally:
        subprocess.run(
            [*compose, "down", "--remove-orphans", "--rmi", "local"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
