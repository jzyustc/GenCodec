"""Public model bundles: config.json, model.pt and entropy_control_int.pt."""
import hashlib
import json
from pathlib import Path

import torch

from codec.config import build_model, load_config
from codec.models.entropy.int8scale import (
    attach_integer_linear_cdf_index_decoder, require_cross_platform_entropy,
)
from codec.models.entropy.meta_prior import MetaPrior
from codec.utils.model_io import atomic_torch_save, load_native_state_dict, sha256

META_PRIOR_BANKS = 64


def save_model(model, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["format_version"] = 2
    config["meta_prior_fitted"] = model.entropy_model.meta_prior is not None
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    atomic_torch_save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        directory / "model.pt",
    )
    return directory / "model.pt"


def load_model(directory, device="cpu", *, entropy=True):
    if entropy:
        from codec.models.entropy.entropy_coding import require_entropy_extension
        require_entropy_extension()
    directory = Path(directory)
    config = load_config(directory / "config.json")
    if config["format_version"] != 2:
        raise ValueError("unsupported model bundle format")
    model = build_model(config)
    if config["meta_prior_fitted"]:
        model.entropy_model.meta_prior = MetaPrior(
            qp_num=8, bank_count=META_PRIOR_BANKS, channel=model.z_ch)
    model.load_state_dict(load_native_state_dict(directory / "model.pt"), strict=True)
    if config["meta_prior_fitted"] and not bool(model.entropy_model.meta_prior.ready.item()):
        raise ValueError("bundle declares a fitted Meta Prior but its state is not ready")
    model.to(device).eval()
    model_hash = sha256(directory / "model.pt")
    if entropy:
        artifact = directory / "entropy_control_int.pt"
        if not artifact.is_file():
            raise FileNotFoundError(
                f"{artifact}: run postprocess.py (Meta Prior and integer CDF export) first"
            )
        attach_integer_linear_cdf_index_decoder(model.entropy_model, artifact)
        expected = getattr(model.entropy_model, "_entropy_control_checkpoint_sha256", None)
        if expected != model_hash:
            raise ValueError("model.pt and entropy_control_int.pt are from different bundles")
        require_cross_platform_entropy(model.entropy_model)
        digest = hashlib.sha256()
        digest.update(model_hash.encode())
        digest.update(sha256(artifact).encode())
        digest.update(json.dumps(config["architecture"], sort_keys=True).encode())
        model.bundle_fingerprint = digest.hexdigest()
    model.release_config = config
    return model
