"""Post-training fitting for the spatially adaptive Meta Prior."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from codec.models.entropy.entropy_models import bit_estimator_z_fwd
from codec.models.entropy.meta_prior import MetaPrior
from codec.training.meta_prior_index_merge import fit_index_merge_model


class _BankFitter(nn.Module):
    def __init__(self, h: Tensor, b: Tensor, a: Tensor):
        super().__init__()
        self.h = nn.Parameter(h)
        self.b = nn.Parameter(b)
        self.a = nn.Parameter(a)

    def probabilities(self, values: Tensor) -> Tensor:
        return bit_estimator_z_fwd(values, self.h, self.b, self.a)


def _deterministic_crop(path: Path, crop_size: int, seed: int) -> Tensor:
    image = Image.open(path).convert("RGB")
    if min(image.size) < crop_size:
        scale = crop_size / min(image.size)
        image = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)), Image.Resampling.BILINEAR)
    rng = random.Random(seed)
    max_left = image.width - crop_size
    max_top = image.height - crop_size
    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0
    image = image.crop((left, top, left + crop_size, top + crop_size))
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().unsqueeze(0) / 255


@torch.no_grad()
def _complexity_components(
    image: Tensor,
    z_height: int,
    z_width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    image = image.float()
    coefficients = image.new_tensor(
        [0.299, 0.587, 0.114]
    )[None, :, None, None]
    luma = (image * coefficients).sum(dim=1, keepdim=True)
    dx = torch.abs(luma[..., 1:] - luma[..., :-1])
    dy = torch.abs(luma[..., 1:, :] - luma[..., :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    gradient = F.adaptive_avg_pool2d(
        0.5 * (dx + dy),
        (z_height, z_width),
    )
    mean = F.adaptive_avg_pool2d(luma, (z_height, z_width))
    mean_square = F.adaptive_avg_pool2d(
        luma.square(),
        (z_height, z_width),
    )
    local_std = (mean_square - mean.square()).clamp_min(0).sqrt()
    coarse_height = max(1, math.ceil(z_height / 2))
    coarse_width = max(1, math.ceil(z_width / 2))
    coarse_mean = F.adaptive_avg_pool2d(
        luma,
        (coarse_height, coarse_width),
    )
    coarse_mean_square = F.adaptive_avg_pool2d(
        luma.square(),
        (coarse_height, coarse_width),
    )
    coarse_std = (
        coarse_mean_square - coarse_mean.square()
    ).clamp_min(0).sqrt()
    coarse_std = F.interpolate(
        coarse_std,
        size=(z_height, z_width),
        mode="bilinear",
        align_corners=False,
    )
    return gradient, local_std, coarse_std


@torch.inference_mode()
def collect_meta_prior_training_data(
    model,
    image_paths,
    *,
    device: torch.device,
    qps=tuple(range(8)),
    qp_batch: int = 4,
    crop_size: int = 512,
    seed: int = 0,
    progress_every: int = 100,
) -> dict:
    """Extract rounded z vectors and spatial initialization statistics."""
    selected_paths = [Path(path) for path in image_paths]
    z_by_qp = {int(qp): [] for qp in qps}
    gradients = []
    local_stds = []
    coarse_stds = []
    started = time.time()
    entropy_model = model.entropy_model
    model.eval()

    for image_index, path in enumerate(selected_paths):
        image = _deterministic_crop(
            path,
            crop_size,
            seed + image_index * 1000003,
        ).to(device=device, dtype=torch.float32)
        z_height = z_width = None
        for qp_start in range(0, len(qps), int(qp_batch)):
            sub_qps = tuple(
                int(qp)
                for qp in qps[qp_start : qp_start + int(qp_batch)]
            )
            repeated = image.expand(
                len(sub_qps),
                -1,
                -1,
                -1,
            ).contiguous()
            qp_index = model._qp_index(
                torch.tensor(sub_qps, device=device),
                len(sub_qps),
                device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=False,
            ):
                q_scale = model._encoder_q_scale(qp_index)
                y = model.encoder(repeated, q_scale)
                q_hyper_z = entropy_model._select_q_scale_bounded(
                    entropy_model.q_scale_hyper_z,
                    qp_index,
                )
                z_hat = entropy_model.hyper_enc(y, q_hyper_z).round()
            _, channel, z_height, z_width = z_hat.shape
            z_cpu = (
                z_hat.permute(0, 2, 3, 1)
                .reshape(len(sub_qps), -1, channel)
                .to(device="cpu", dtype=torch.int16)
                .numpy()
            )
            for batch_index, qp in enumerate(sub_qps):
                z_by_qp[qp].append(z_cpu[batch_index])
        gradient, local_std, coarse_std = _complexity_components(
            image,
            int(z_height),
            int(z_width),
        )
        gradients.append(gradient.flatten().cpu().numpy())
        local_stds.append(local_std.flatten().cpu().numpy())
        coarse_stds.append(coarse_std.flatten().cpu().numpy())
        done = image_index + 1
        if done == 1 or done % progress_every == 0 or done == len(selected_paths):
            print(
                f"[Meta Prior] extracted {done}/{len(selected_paths)} images "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )

    return {
        "z_by_qp": {
            qp: np.concatenate(values, axis=0)
            for qp, values in z_by_qp.items()
        },
        "gradient": np.concatenate(gradients),
        "local_std": np.concatenate(local_stds),
        "coarse_std": np.concatenate(coarse_stds),
        "num_images": len(selected_paths),
        "crop_size": int(crop_size),
        "seed": int(seed),
        "z_height": int(z_height),
        "z_width": int(z_width),
    }


def _complexity_score(data: dict) -> tuple[np.ndarray, np.ndarray]:
    gradient = data["gradient"]
    local_std = data["local_std"]
    coarse_std = data["coarse_std"]
    scales = np.maximum(
        np.asarray(
            [
                np.quantile(gradient, 0.75),
                np.quantile(local_std, 0.75),
                np.quantile(coarse_std, 0.75),
            ]
        ),
        1e-8,
    )
    score = (
        0.50 * np.clip(gradient / scales[0], 0, 4)
        + 0.30 * np.clip(local_std / scales[1], 0, 4)
        + 0.20 * np.clip(coarse_std / scales[2], 0, 4)
    )
    return score, scales


def _initial_assignments(score: np.ndarray, bank_count: int) -> np.ndarray:
    edges = np.quantile(
        score,
        np.arange(1, bank_count, dtype=np.float64) / bank_count,
    )
    return np.searchsorted(edges, score, side="right").astype(np.int16)


def _histogram_counts(
    z: np.ndarray,
    assignments: np.ndarray,
    bank_count: int,
    radius: int,
) -> np.ndarray:
    if int(z.min()) < -radius or int(z.max()) > radius:
        raise RuntimeError(
            f"z range [{z.min()}, {z.max()}] exceeds ±{radius}"
        )
    _, channel = z.shape
    symbol_count = 2 * radius + 1
    counts = np.zeros(
        (bank_count, channel, symbol_count),
        dtype=np.float64,
    )
    shifted = z.astype(np.int64) + radius
    for bank in range(bank_count):
        selected = assignments == bank
        if not bool(selected.any()):
            # Hard-EM can leave banks unused; retain their previous parameters.
            continue
        values = shifted[selected]
        for channel_index in range(channel):
            counts[bank, channel_index] = np.bincount(
                values[:, channel_index],
                minlength=symbol_count,
            )
    return counts


def _fit_counts(
    prior: _BankFitter,
    counts_np: np.ndarray,
    values: Tensor,
    *,
    steps: int,
    lr: float,
) -> list[float]:
    counts = torch.as_tensor(
        counts_np,
        device=values.device,
        dtype=torch.float32,
    )
    normalization = counts.sum(dim=(1, 2))
    active = normalization > 0
    if not bool(active.any()):
        raise ValueError("Meta Prior fitting requires at least one hyperlatent")
    normalization = normalization.clamp_min(1)
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=lr * 0.02,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probability = prior.probabilities(values)[:, :, :, 0].float()
        bits = -torch.log2(probability.clamp_min(1e-9))
        cross_entropy = (
            (bits * counts).sum(dim=(1, 2)) / normalization
        )
        cross_entropy[active].mean().backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        probability = prior.probabilities(values)[:, :, :, 0].float()
        bits = -torch.log2(probability.clamp_min(1e-9))
        cross_entropy = (
            (bits * counts).sum(dim=(1, 2)) / normalization
        )
    return cross_entropy.cpu().tolist()


@torch.no_grad()
def _assign_patches(
    prior: _BankFitter,
    z: np.ndarray,
    values: Tensor,
    *,
    radius: int,
    chunk: int,
) -> tuple[np.ndarray, float]:
    probability = prior.probabilities(values)[:, :, :, 0].float()
    cost_table = -torch.log2(probability.clamp_min(1e-9))
    assignments = np.empty(len(z), dtype=np.int16)
    total_bits = 0.0
    for start in range(0, len(z), chunk):
        stop = min(start + chunk, len(z))
        indexes = torch.as_tensor(
            z[start:stop].astype(np.int64) + radius,
            device=values.device,
            dtype=torch.long,
        )
        gathered = torch.gather(
            cost_table[:, None].expand(
                -1,
                stop - start,
                -1,
                -1,
            ),
            3,
            indexes[None, :, :, None].expand(
                cost_table.shape[0],
                -1,
                -1,
                1,
            ),
        ).squeeze(3)
        best_cost, best_bank = gathered.sum(dim=2).min(dim=0)
        assignments[start:stop] = (
            best_bank.to(device="cpu", dtype=torch.int16).numpy()
        )
        total_bits += float(best_cost.sum())
    return assignments, total_bits / (len(z) * z.shape[1])


def _fit_one_qp(
    base_prior,
    z: np.ndarray,
    score: np.ndarray,
    *,
    qp: int,
    bank_count: int,
    device: torch.device,
    em_iters: int,
    first_steps: int,
    later_steps: int,
    lr: float,
    symbol_radius: int,
    assign_chunk: int,
) -> tuple[dict, dict]:
    h0, b0, a0 = base_prior._select(qp)
    prior = _BankFitter(
        h0.detach().repeat(bank_count, 1, 1),
        b0.detach().repeat(bank_count, 1, 1),
        a0.detach().repeat(bank_count, 1, 1),
    ).to(device)
    symbols = torch.arange(
        -symbol_radius,
        symbol_radius + 1,
        device=device,
        dtype=torch.float32,
    )
    values = symbols.view(1, 1, -1, 1).expand(
        bank_count,
        z.shape[1],
        -1,
        1,
    )
    assignments = _initial_assignments(score, bank_count)
    history = []
    started = time.time()
    for iteration in range(em_iters):
        counts = _histogram_counts(
            z,
            assignments,
            bank_count,
            symbol_radius,
        )
        steps = first_steps if iteration == 0 else later_steps
        cluster_ce = _fit_counts(
            prior,
            counts,
            values,
            steps=steps,
            lr=lr,
        )
        updated, selected_ce = _assign_patches(
            prior,
            z,
            values,
            radius=symbol_radius,
            chunk=assign_chunk,
        )
        changed = float(np.mean(updated != assignments))
        populations = np.bincount(updated, minlength=bank_count)
        history.append(
            {
                "iteration": iteration,
                "fit_steps": steps,
                "cluster_ce_bits_per_symbol": cluster_ce,
                "selected_ce_bits_per_symbol": selected_ce,
                "assignment_changed_fraction": changed,
                "populations": populations.tolist(),
            }
        )
        print(
            f"[Meta Prior] QP{qp} EM{iteration}: "
            f"{selected_ce:.6f} bit/symbol, "
            f"changed={changed * 100:.2f}%, "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        assignments = updated

    counts = _histogram_counts(
        z,
        assignments,
        bank_count,
        symbol_radius,
    )
    final_cluster_ce = _fit_counts(
        prior,
        counts,
        values,
        steps=first_steps if em_iters == 0 else later_steps,
        lr=lr,
    )
    final_assignments, final_selected_ce = _assign_patches(
        prior,
        z,
        values,
        radius=symbol_radius,
        chunk=assign_chunk,
    )
    metadata = {
        "qp": int(qp),
        "bank_count": int(bank_count),
        "selector_bits_per_position": int(math.ceil(math.log2(bank_count))),
        "num_patches": int(len(z)),
        "channels": int(z.shape[1]),
        "em_iters": int(em_iters),
        "first_steps": int(first_steps),
        "later_steps": int(later_steps),
        "lr": float(lr),
        "symbol_radius": int(symbol_radius),
        "history": history,
        "final_cluster_ce_bits_per_symbol": final_cluster_ce,
        "final_selected_ce_bits_per_symbol": final_selected_ce,
        "final_populations": np.bincount(
            final_assignments,
            minlength=bank_count,
        ).tolist(),
    }
    return (
        {
            "h": prior.h.detach().cpu(),
            "b": prior.b.detach().cpu(),
            "a": prior.a.detach().cpu(),
        },
        metadata,
    )


def fit_meta_prior(
    model,
    data: dict,
    *,
    device: torch.device,
    bank_count: int = 64,
    em_iters: int = 6,
    first_steps: int = 500,
    later_steps: int = 300,
    lr: float = 0.01,
    symbol_radius: int = 127,
    assign_chunk: int = 8192,
    index_merge: bool = True,
) -> dict:
    """Fit all QPs and return one checkpoint-ready Meta Prior payload."""
    score, scales = _complexity_score(data)
    per_qp = []
    parameters = {"h": [], "b": [], "a": []}
    for qp in sorted(data["z_by_qp"]):
        fitted, metadata = _fit_one_qp(
            model.entropy_model.entropy_bottleneck,
            data["z_by_qp"][qp],
            score,
            qp=qp,
            bank_count=bank_count,
            device=device,
            em_iters=em_iters,
            first_steps=first_steps,
            later_steps=later_steps,
            lr=lr,
            symbol_radius=symbol_radius,
            assign_chunk=assign_chunk,
        )
        for name in parameters:
            parameters[name].append(fitted[name])
        per_qp.append(metadata)
    payload = {
        "format_version": 2,
        "algorithm": "Meta Prior",
        "selection": "minimum_code_length",
        "bank_count": int(bank_count),
        "selector_bits_per_position": int(math.ceil(math.log2(bank_count))),
        "selector_coding": "Meta-Prior Index Merge Coding",
        "h": torch.stack(parameters["h"]),
        "b": torch.stack(parameters["b"]),
        "a": torch.stack(parameters["a"]),
        "complexity_initialization_scales": scales.tolist(),
        "extraction": {
            "num_images": data["num_images"],
            "crop_size": data["crop_size"],
            "seed": data["seed"],
        },
        "per_qp": per_qp,
    }
    if not index_merge:
        payload["format_version"] = 1
        payload["selector_coding"] = "Meta-Prior Index Merge Coding (uniform initialization)"
        return payload
    temporary = MetaPrior(
        qp_num=len(parameters["h"]),
        bank_count=bank_count,
        channel=int(data["z_by_qp"][min(data["z_by_qp"])].shape[1]),
    ).to(device)
    temporary.load_fitted(payload)
    from codec.models.entropy import entropy_coding as ec

    z_height = int(data.get("z_height", 0))
    z_width = int(data.get("z_width", 0))
    num_images = int(data["num_images"])
    if z_height <= 0 or z_width <= 0:
        positions_per_image = (
            len(data["z_by_qp"][min(data["z_by_qp"])]) // num_images
        )
        inferred = int(round(math.sqrt(positions_per_image)))
        if inferred * inferred != positions_per_image:
            raise ValueError(
                "cannot infer Meta Prior selector-map shape for Index Merge"
            )
        z_height = z_width = inferred
    records_by_qp = {}
    for qp in sorted(data["z_by_qp"]):
        values = data["z_by_qp"][qp]
        expected = num_images * z_height * z_width
        if len(values) != expected:
            raise ValueError(
                f"QP{qp} z record count {len(values)} != {expected}"
            )
        maps = values.reshape(
            num_images,
            z_height,
            z_width,
            values.shape[1],
        )
        cdf, lengths = ec.update_meta_prior_cdf(temporary, int(qp))
        selected_maps = []
        for start in range(0, num_images, 32):
            z = torch.as_tensor(
                maps[start : start + 32],
                device=device,
                dtype=torch.int16,
            ).permute(0, 3, 1, 2)
            selected = ec.select_meta_prior_indexes_from_cdf(
                z,
                cdf,
                lengths,
                bank_count=bank_count,
            )
            selected_maps.extend(selected.cpu().numpy())
        records_by_qp[int(qp)] = selected_maps
    payload["index_merge_model"] = fit_index_merge_model(
        records_by_qp,
        qp_num=len(parameters["h"]),
        bank_count=bank_count,
    )
    return payload


def save_meta_prior_artifact(payload: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"h", "b", "a"}
    }
    index_merge = metadata.get("index_merge_model")
    if index_merge is not None:
        index_merge = dict(index_merge)
        probabilities = torch.as_tensor(
            index_merge.pop("probabilities"),
            dtype=torch.int32,
        )
        index_merge["probability_shape"] = list(probabilities.shape)
        index_merge["probability_min"] = int(probabilities.min().item())
        index_merge["probability_max"] = int(probabilities.max().item())
        metadata["index_merge_model"] = index_merge
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
