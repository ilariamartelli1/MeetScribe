import os
import subprocess
import sys


def check_device(device: str) -> None:
    if device == "cpu":
        return
    if device != "cuda":
        raise ValueError("MEETSCRIBE_DEVICE must be 'cuda' or 'cpu'.")

    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RuntimeError(
            "NVIDIA GPU preflight failed. Check the host driver and NVIDIA Container "
            "Toolkit, or explicitly use the CPU test configuration."
        ) from exc


def main() -> None:
    try:
        check_device(os.environ.get("MEETSCRIBE_DEVICE", ""))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    os.execvp(
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )


if __name__ == "__main__":
    main()
