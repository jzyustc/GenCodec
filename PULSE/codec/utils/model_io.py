"""Tensor-only checkpoint IO."""
from pathlib import Path
import hashlib
import os
import tempfile

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_native_state_dict(path):
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    return value["state_dict"]


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
