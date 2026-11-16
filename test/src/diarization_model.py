"""Verify the provisioned, pinned Community-1 artifacts before loading."""

import hashlib
from pathlib import Path

MODEL_FILES = {
    "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
    "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}


def verify_model(path: Path) -> None:
    for name, digest in MODEL_FILES.items():
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f"Missing Community-1 model file: {file}")
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(
                    f"Community-1 model file failed SHA-256 verification: {file}"
                )
