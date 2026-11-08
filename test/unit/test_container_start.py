"""Verify device preflight, startup errors, and container server launch."""

import subprocess
from unittest.mock import Mock

import pytest

from app.container_start import check_device, main


def test_cpu_mode_does_not_run_gpu_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    check_device("cpu")

    run.assert_not_called()


def test_cuda_mode_checks_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    check_device("cuda")

    run.assert_called_once_with(
        ["nvidia-smi", "-L"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(1, ["nvidia-smi", "-L"]),
        subprocess.TimeoutExpired(["nvidia-smi", "-L"], 10),
    ],
)
def test_cuda_preflight_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))

    with pytest.raises(RuntimeError, match="NVIDIA GPU preflight failed"):
        check_device("cuda")


@pytest.mark.parametrize("device", ["", "gpu", "CPU"])
def test_invalid_device_is_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="MEETSCRIBE_DEVICE"):
        check_device(device)


def test_main_starts_server_in_cpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    execvp = Mock()
    monkeypatch.setenv("MEETSCRIBE_DEVICE", "cpu")
    monkeypatch.setattr("os.execvp", execvp)

    main()

    execvp.assert_called_once_with(
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )


def test_main_exits_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MEETSCRIBE_DEVICE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "MEETSCRIBE_DEVICE" in capsys.readouterr().err
