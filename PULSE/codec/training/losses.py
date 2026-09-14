"""Paper objectives: MSE; LPIPS pre-training; DINO adversarial and ROI losses."""
from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.attention import SDPBackend, sdpa_kernel

from codec.losses.semantic_roi import SemanticROIFeatureLoss


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_weight=1.0, alex_weight=0.5, *, roi=False):
        super().__init__()
        import lpips
        self.vgg = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
        self.alex = lpips.LPIPS(net="alex").eval().requires_grad_(False)
        self.vgg_weight, self.alex_weight = vgg_weight, alex_weight
        # ROI reuses LPIPS for full images and conditional crops in one backward.
        # Keep Inductor compilation, without CUDA Graph pool ownership tracking.
        compile_options = {"options": {"triton.cudagraphs": False}} if roi else {"mode": "reduce-overhead"}
        self.vgg = torch.compile(self.vgg, dynamic=False, **compile_options)
        self.alex = torch.compile(self.alex, dynamic=False, **compile_options)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, reconstruction, target):
        vgg = self.vgg(reconstruction, target).flatten(1).mean(dim=1)
        alex = self.alex(reconstruction, target).flatten(1).mean(dim=1)
        return self.vgg_weight * vgg + self.alex_weight * alex


class DinoDiscriminator(nn.Module):
    def __init__(self, latent_channels, source="facebookresearch/dinov2"):
        super().__init__()
        self.backbone = torch.hub.load(
            source, "dinov2_vitb14", source="local" if Path(source).is_dir() else "github",
            trust_repo=True,
        ).to(torch.bfloat16).eval().requires_grad_(False)
        dim = self.backbone.embed_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(2 * dim, dim, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(dim, dim // 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(dim // 2, 1, 1)),
            ) for _ in range(4)
        ])
        self.cond_proj = nn.ModuleList([
            spectral_norm(nn.Conv2d(latent_channels, dim, 1)) for _ in range(4)
        ])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image, latent):
        # Match the paper's /16 feature grid using DINOv2's patch-14 backbone.
        h, w = image.shape[-2:]
        image = F.interpolate(image, (14 * (h // 16), 14 * (w // 16)),
                              mode="bicubic", align_corners=False)
        # cuDNN SDPA in torch 2.5 can produce NaN input gradients under BF16.
        # Keep gradients through the frozen teacher, using the other backends.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.MATH]):
            features = self.backbone.get_intermediate_layers(
                image, n=(2, 5, 8, 11), reshape=True, return_class_token=False
            )
        logits = []
        for feature, proj, head in zip(features, self.cond_proj, self.heads):
            feature = feature.contiguous().to(torch.bfloat16)
            condition = F.interpolate(latent.detach(), feature.shape[-2:], mode="bilinear", align_corners=False)
            condition = proj(condition.to(feature.dtype))
            logits.append(head(torch.cat([feature, condition], dim=1)))
        return torch.cat(logits, dim=1)


def discriminator_loss(real, fake):
    """Apply hinge to every local logit before reduction, as in the source."""
    return 0.5 * (F.relu(1.0 + fake).mean() + F.relu(1.0 - real).mean())


def roi_lpips(perceptual, reconstruction, target, mask, patch_size):
    """Sample a fixed-size crop around a uniformly chosen ROI-mask pixel."""
    valid = mask.sum(dim=(1, 2, 3)) > 0
    if not bool(valid.any()):
        return reconstruction.new_zeros(reconstruction.shape[0])
    h, w = reconstruction.shape[-2:]
    side = min(patch_size, h, w)
    rec, gt = [], []
    for i in range(reconstruction.shape[0]):
        if bool(valid[i]):
            points = torch.nonzero(mask[i, 0] > 0, as_tuple=False)
            point = torch.randint(len(points), (1,), device=points.device).item()
            cy, cx = points[point].tolist()
        else:
            cy, cx = h // 2, w // 2
        top = max(0, min(h - side, round(cy - side / 2)))
        left = max(0, min(w - side, round(cx - side / 2)))
        rec.append(reconstruction[i:i + 1, :, top:top + side, left:left + side])
        gt.append(target[i:i + 1, :, top:top + side, left:left + side])
    values = perceptual(torch.cat(rec), torch.cat(gt))
    return values * valid.to(values.dtype)


class RDObjective(nn.Module):
    def __init__(self, settings, qp_num, *, face_teacher=None, ocr_teacher=None):
        super().__init__()
        self.settings = settings
        self.register_buffer("lambdas", torch.exp(torch.linspace(
            math.log(settings["lambda_min"]), math.log(settings["lambda_max"]), qp_num)))
        self.register_buffer("lambda_geomean", torch.sqrt(self.lambdas[0] * self.lambdas[-1]))
        weight = settings.get("lpips", 0)
        self.perceptual = PerceptualLoss(
            weight, settings.get("lpips_alex", weight * .5), roi=settings.get("roi", False)) if weight else None
        self.semantic = None
        if settings.get("roi", False):
            if not face_teacher or not ocr_teacher:
                raise ValueError("Stage II requires --face-teacher and --ocr-teacher")
            self.semantic = SemanticROIFeatureLoss(
                face_model_path=face_teacher, ocr_model_path=ocr_teacher,
                face_context_scale=settings.get("face_context_scale", 1.5),
                ocr_context_scale=settings.get("ocr_context_scale", 1.1))

    def forward(self, reconstruction, image, rates, qp, metadata, gan=None):
        target = image * 2 - 1
        mse = ((reconstruction - target) ** 2).mean(dim=(1, 2, 3))
        lam = self.lambdas[qp]
        mse_weight = self.settings.get("mse", 1.0) * (lam / self.lambda_geomean) ** self.settings.get("mse_alpha", 0.0)
        metrics = {"mse": mse.mean().detach()}
        if self.perceptual is not None:
            perceptual = self.perceptual(reconstruction, target)
            distortion = (self.settings["mse"] if self.semantic is not None else mse_weight) * mse + perceptual
            metrics["lpips"] = perceptual.mean().detach()
        else:
            distortion = mse_weight * mse
        if self.semantic is not None:
            mask = metadata["roi_mask"].to(device=reconstruction.device, dtype=reconstruction.dtype).clamp(0, 1)
            mass = mask.sum(dim=(1, 2, 3))
            roi_mse = (((reconstruction - target).pow(2).mean(dim=1, keepdim=True) * mask)
                       .sum(dim=(1, 2, 3)) / mass.clamp_min(1)) * (mass > 0)
            roi_perceptual = roi_lpips(self.perceptual, reconstruction, target, mask,
                                       self.settings.get("roi_lpips_patch_size", 128))
            distortion = (distortion + self.settings.get("roi_mse", .125) * roi_mse
                          + self.settings.get("roi_lpips", .125) * roi_perceptual)
            face, ocr, _, _ = self.semantic(reconstruction, target, metadata)
            distortion = (distortion + self.settings.get("face", .0125) * face
                          + self.settings.get("text", .025) * ocr)
            metrics.update(face=face.mean().detach(), text=ocr.mean().detach(),
                           roi_mse=roi_mse.mean().detach())
        # Retain the source's unit distortion scaling operation as well as
        # its reduction order, so BF16 gradient accumulation stays identical.
        distortion = 1.0 * distortion
        distortion_term = (lam * distortion).mean()
        rate_term = rates["sq_loss"].mean()
        if gan is not None:
            gan = gan() if callable(gan) else gan
            gan_objective = (lam * gan).mean()
            loss = distortion_term + rate_term + self.settings.get("gan", 0) * gan_objective
            metrics["gan"] = gan.mean().detach()
        else:
            loss = distortion_term + rate_term
        metrics.update(loss=loss.detach(), bpp=rates["sq_loss"].mean().detach())
        return loss, metrics
