"""Frozen face/OCR feature alignment on synchronized ROI boxes."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticROIFeatureLoss(nn.Module):
    """Align frozen semantic features on matching reconstruction/target crops.

    The face model consumes RGB ``[-1, 1]`` crops and returns one embedding per
    image.  The OCR model consumes BGR ``[-1, 1]`` crops and returns a spatial
    feature sequence shaped ``[B, C, 1, T]``.  Models are exported TorchScript
    modules so training has no runtime dependency on their source packages.
    """

    def __init__(
        self,
        face_model_path=None,
        ocr_model_path=None,
        face_crop_size=160,
        face_context_scale=1.5,
        face_min_crop_side=16.0,
        ocr_crop_height=32,
        ocr_crop_width=100,
        ocr_context_scale=1.1,
        ocr_min_crop_side=8.0,
    ):
        super().__init__()
        self.face_crop_size = int(face_crop_size)
        self.face_context_scale = float(face_context_scale)
        self.face_min_crop_side = float(face_min_crop_side)
        self.ocr_crop_height = int(ocr_crop_height)
        self.ocr_crop_width = int(ocr_crop_width)
        self.ocr_context_scale = float(ocr_context_scale)
        self.ocr_min_crop_side = float(ocr_min_crop_side)
        if self.face_crop_size < 32:
            raise ValueError("face_crop_size must be >= 32")
        if self.ocr_crop_height < 8 or self.ocr_crop_width < 8:
            raise ValueError("OCR crop dimensions must be >= 8")
        if self.face_context_scale < 1 or self.ocr_context_scale < 1:
            raise ValueError("semantic ROI context scales must be >= 1")

        self.face_model = self._load_frozen(face_model_path)
        self.ocr_model = self._load_frozen(ocr_model_path)
        if self.face_model is None and self.ocr_model is None:
            raise ValueError("at least one semantic feature model is required")

    @staticmethod
    def _load_frozen(path):
        if path is None:
            return None
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        model = torch.jit.load(str(path), map_location="cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def train(self, mode=True):
        # Frozen feature networks contain BatchNorm/Dropout and must stay in
        # inference mode even when Lightning recursively calls model.train().
        super().train(False)
        if self.face_model is not None:
            self.face_model.eval()
        if self.ocr_model is not None:
            self.ocr_model.eval()
        return self

    @staticmethod
    def _crop_with_grid(
        images,
        boxes,
        present,
        output_height,
        output_width,
        context_scale,
        min_crop_side,
        square,
    ):
        """Vectorized differentiable crop-and-resize with border padding."""
        batch, _, image_height, image_width = images.shape
        boxes = boxes.to(device=images.device, dtype=torch.float32)
        present = present.to(device=images.device).bool().reshape(batch)
        if boxes.shape != (batch, 4):
            raise ValueError(f"expected boxes [B,4], got {tuple(boxes.shape)}")

        x0, y0, x1, y1 = boxes.unbind(dim=1)
        # Dataset boxes use half-open pixel boundaries [x0, x1), so the first
        # and last sampled pixel centers are x0 and x1-1 respectively.
        center_x = 0.5 * (x0 + x1 - 1)
        center_y = 0.5 * (y0 + y1 - 1)
        width = (x1 - x0).clamp_min(1) * context_scale
        height = (y1 - y0).clamp_min(1) * context_scale
        if square:
            side = torch.maximum(width, height).clamp_min(min_crop_side)
            side = side.clamp_max(float(min(image_height, image_width)))
            width = height = side
        else:
            width = width.clamp(min=min_crop_side, max=float(image_width))
            height = height.clamp(min=min_crop_side, max=float(image_height))

        # Invalid samples retain a stable center crop so feature-network batch
        # shapes do not vary. Their per-sample losses are zeroed below.
        fallback_width = min(float(image_width), max(min_crop_side, 32.0))
        fallback_height = min(float(image_height), max(min_crop_side, 32.0))
        center_x = torch.where(
            present,
            center_x,
            center_x.new_full(center_x.shape, (image_width - 1) / 2),
        )
        center_y = torch.where(
            present,
            center_y,
            center_y.new_full(center_y.shape, (image_height - 1) / 2),
        )
        width = torch.where(present, width, width.new_full(width.shape, fallback_width))
        height = torch.where(present, height, height.new_full(height.shape, fallback_height))

        half_width = 0.5 * (width - 1).clamp_min(0)
        half_height = 0.5 * (height - 1).clamp_min(0)
        center_x = torch.maximum(center_x, half_width)
        center_x = torch.minimum(
            center_x,
            center_x.new_tensor(float(image_width - 1)) - half_width,
        )
        center_y = torch.maximum(center_y, half_height)
        center_y = torch.minimum(
            center_y,
            center_y.new_tensor(float(image_height - 1)) - half_height,
        )

        sample_x = torch.linspace(
            -1, 1, output_width, device=images.device, dtype=torch.float32
        ).view(1, 1, output_width)
        sample_y = torch.linspace(
            -1, 1, output_height, device=images.device, dtype=torch.float32
        ).view(1, output_height, 1)
        pixel_x = center_x[:, None, None] + sample_x * half_width[:, None, None]
        pixel_y = center_y[:, None, None] + sample_y * half_height[:, None, None]
        grid_x = pixel_x * (2.0 / max(image_width - 1, 1)) - 1.0
        grid_y = pixel_y * (2.0 / max(image_height - 1, 1)) - 1.0
        grid = torch.stack(
            (
                grid_x.expand(-1, output_height, -1),
                grid_y.expand(-1, -1, output_width),
            ),
            dim=-1,
        )
        crops = F.grid_sample(
            images.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return crops.to(images.dtype), present

    @staticmethod
    def _masked(values, valid):
        return values * valid.to(values.dtype)

    @staticmethod
    def _feature_forward(model, crops):
        """Run frozen teachers in FP32, even inside mixed-precision training.

        The codec reconstruction is BF16 under the Stage-2 autocast context.
        Letting TorchScript BatchNorm participate in that context can build a
        BF16 forward whose backward receives an FP32 gradient from the cosine
        objective.  CUDA then fails with ``Expected grad_output ... BFloat16``.
        Keeping the small ROI teacher pass in FP32 avoids that TorchScript
        autocast edge case while preserving gradients back to ``crops``.
        """
        with torch.autocast(device_type=crops.device.type, enabled=False):
            return model(crops.float()).float()

    def face_loss(self, reconstruction, target, metadata):
        if self.face_model is None:
            return reconstruction.new_zeros(reconstruction.shape[0]), None
        valid = metadata["face_present"].to(reconstruction.device).bool().flatten()
        # Semantic objectives are deliberately FP32 even when the codec output
        # is BF16.  This also keeps index_copy dtype-stable after valid-sample
        # compaction.
        result = torch.zeros(
            reconstruction.shape[0], device=reconstruction.device, dtype=torch.float32
        )
        if not bool(valid.any()):
            return result, valid
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        selected_boxes = metadata["face_box"].to(reconstruction.device)[indices]
        recon_crop, valid = self._crop_with_grid(
            reconstruction[indices],
            selected_boxes,
            torch.ones(len(indices), device=reconstruction.device),
            self.face_crop_size,
            self.face_crop_size,
            self.face_context_scale,
            self.face_min_crop_side,
            square=True,
        )
        target_crop, _ = self._crop_with_grid(
            target[indices],
            selected_boxes,
            torch.ones(len(indices), device=reconstruction.device),
            self.face_crop_size,
            self.face_crop_size,
            self.face_context_scale,
            self.face_min_crop_side,
            square=True,
        )
        recon_features = self._feature_forward(self.face_model, recon_crop)
        with torch.no_grad():
            target_features = self._feature_forward(self.face_model, target_crop)
        values = 1 - F.cosine_similarity(recon_features, target_features, dim=-1)
        return result.index_copy(0, indices, values), metadata["face_present"].to(
            reconstruction.device
        ).bool().flatten()

    def ocr_loss(self, reconstruction, target, metadata):
        if self.ocr_model is None:
            return reconstruction.new_zeros(reconstruction.shape[0]), None
        valid = metadata["text_present"].to(reconstruction.device).bool().flatten()
        result = torch.zeros(
            reconstruction.shape[0], device=reconstruction.device, dtype=torch.float32
        )
        if not bool(valid.any()):
            return result, valid
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        selected_boxes = metadata["text_box"].to(reconstruction.device)[indices]
        recon_crop, valid = self._crop_with_grid(
            reconstruction[indices],
            selected_boxes,
            torch.ones(len(indices), device=reconstruction.device),
            self.ocr_crop_height,
            self.ocr_crop_width,
            self.ocr_context_scale,
            self.ocr_min_crop_side,
            square=False,
        )
        target_crop, _ = self._crop_with_grid(
            target[indices],
            selected_boxes,
            torch.ones(len(indices), device=reconstruction.device),
            self.ocr_crop_height,
            self.ocr_crop_width,
            self.ocr_context_scale,
            self.ocr_min_crop_side,
            square=False,
        )
        # The OpenCV CRNN was exported with blobFromImage(swapRB=False), so it
        # consumes normalized BGR rather than the codec's RGB channel order.
        recon_features = self._feature_forward(
            self.ocr_model, recon_crop[:, [2, 1, 0]]
        )
        with torch.no_grad():
            target_features = self._feature_forward(
                self.ocr_model, target_crop[:, [2, 1, 0]]
            )
        recon_features = F.normalize(recon_features, dim=1, eps=1e-6)
        target_features = F.normalize(target_features, dim=1, eps=1e-6)
        values = (1 - (recon_features * target_features).sum(dim=1)).flatten(1).mean(dim=1)
        return result.index_copy(0, indices, values), metadata["text_present"].to(
            reconstruction.device
        ).bool().flatten()

    def forward(self, reconstruction, target, metadata):
        face_values, face_valid = self.face_loss(reconstruction, target, metadata)
        ocr_values, text_valid = self.ocr_loss(reconstruction, target, metadata)
        return face_values, ocr_values, face_valid, text_valid
