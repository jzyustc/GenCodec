"""Build and validate a complete postprocessed checkpoint before publishing it."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import torch

from codec.checkpoint import load_model, META_PRIOR_BANKS
from codec import bitstream
from codec.runtime import configure_runtime, compress_image, decompress_image
from codec.utils.model_io import sha256

BUNDLE_FILES = ("config.json", "model.pt", "entropy_control_int.pt")


@contextmanager
def new_bundle_output(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint output must be a new directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.postprocess-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        yield staged
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint output appeared during processing: {destination}")
        staged.rename(destination)


def write_bundle_checksums(directory):
    directory = Path(directory)
    text = "".join(f"{sha256(directory / name)}  {name}\n" for name in BUNDLE_FILES)
    handle, temporary = tempfile.mkstemp(prefix=".SHA256SUMS.", dir=directory)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(temporary, directory / "SHA256SUMS")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.inference_mode()
def validate_bundle(directory, image, *, device="cpu"):
    configure_runtime(device, torch.get_num_threads())
    model = load_model(directory, device)
    entropy = model.entropy_model
    if entropy.meta_prior is None or entropy.meta_prior.bank_count != META_PRIOR_BANKS:
        raise ValueError("postprocessed checkpoint requires a fitted 64-bank Meta Prior")
    expected_mode = "part_specific"
    if entropy._y_cdf_mode != expected_mode:
        raise ValueError(f"postprocessed checkpoint requires {expected_mode} y CDFs")
    if model.release_config["coding"]["skip_index_cutoff"] != 2:
        raise ValueError("postprocessed checkpoint requires skip-index cutoff 2")
    image = image.to(device)
    for qp in range(model.qp_num):
        stream, _ = compress_image(model, image, qp)
        metadata, payload = bitstream.unpack(stream)
        if metadata["y_cdf_mode"] != expected_mode or metadata["skip_index_cutoff"] != 2 or payload[:5] != b"MPRI\x02":
            raise RuntimeError(f"postprocessed checkpoint has incorrect QP{qp} coding defaults")
        decoded, _ = decompress_image(model, stream)
        if decoded.shape != image.shape or not torch.isfinite(decoded).all():
            raise RuntimeError(f"postprocessed checkpoint failed QP{qp} roundtrip")
    write_bundle_checksums(directory)
