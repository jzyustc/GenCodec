"""End-to-end compression, independent decompression and timing helpers."""
import time

import torch

from codec import bitstream
from codec.images import pad_image
from codec.models.entropy.y_cdf import resolve_y_cdf


def configure_runtime(device="cpu", threads=1):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def compress_image(model, image, qp=3, *, legacy=False):
    if image.ndim != 4 or image.shape[0:2] != (1, 3) or min(image.shape[-2:]) < 1:
        raise ValueError("expected one RGB image with shape [1,3,H,W]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("input pixels must be finite and in [0,1]")
    if not 0 <= qp < model.qp_num:
        raise ValueError(f"QP must be in [0,{model.qp_num - 1}]")
    h, w = image.shape[-2:]
    if h * w > bitstream.MAX_PIXELS or max(h, w) > 65535:
        raise ValueError("image is too large")
    synchronize(image.device)
    start = time.perf_counter()
    if not model.entropy_model.meta_prior_enabled:
        raise ValueError("inference requires a fitted Meta Prior; run postprocess.py first")
    result = model.compress(pad_image(image), qp=qp, skip_index_cutoff=2,
                            y_cdf_mode="part_specific")
    synchronize(image.device)
    data = bitstream.pack(
        dict(height=h, width=w, qp=int(result["qp"]),
             y_shape=list(result["y_shape"]), z_shape=list(result["z_shape"]),
             y_cdf_mode=result.get("y_cdf_mode", "gaussian"),
             skip_index_cutoff=int(result.get("skip_index_cutoff", -1))),
        bytes(result["stream"]),
        fingerprint=model.bundle_fingerprint, legacy=legacy,
    )
    return data, 1000 * (time.perf_counter() - start)


@torch.inference_mode()
def decompress_image(model, data, *, allow_legacy=False):
    device = next(model.parameters()).device
    synchronize(device)
    start = time.perf_counter()
    metadata, payload = bitstream.unpack(data, allow_legacy=allow_legacy)
    if metadata["model_id"] is not None and metadata["model_id"] != bytes.fromhex(model.bundle_fingerprint)[:16]:
        raise ValueError("bitstream was encoded with a different model bundle")
    h, w = metadata["height"], metadata["width"]
    ph, pw = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    expected_y = [1, model.M, ph // 16, pw // 16]
    expected_z = [1, model.z_ch, ph // 64, pw // 64]
    if metadata["y_shape"] != expected_y or metadata["z_shape"] != expected_z:
        raise ValueError("latent shapes do not match image dimensions and model")
    # A decoder follows the stream, including older Gaussian/non-skip files.
    resolve_y_cdf(model.entropy_model, metadata["y_cdf_mode"])
    latent = model.decompress(payload, metadata["z_shape"], metadata["y_shape"],
                              qp=metadata["qp"], skip_index_cutoff=None if metadata["skip_index_cutoff"] < 0 else metadata["skip_index_cutoff"],
                              y_cdf_mode=metadata["y_cdf_mode"],
                              entropy_lanes=metadata.get("entropy_lanes", 1),
                              entropy_sections=metadata.get("entropy_sections"),
                              entropy_dense=metadata.get("entropy_dense", False))
    image = model.decode(latent, qp=metadata["qp"])[..., :h, :w]
    synchronize(device)
    return image, 1000 * (time.perf_counter() - start)
