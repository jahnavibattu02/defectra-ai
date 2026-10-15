from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(__import__("os").environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
IMG_SIZE = 256


class ConvAutoencoder(nn.Module):
    """Architecture copied from the user's notebook; keep it in sync with training."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _png_base64(array: np.ndarray) -> str:
    """Encode a normalized 2D heatmap as a small grayscale PNG."""
    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(array.min()), float(array.max())
    if hi > lo:
        scaled = (array - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(array)
    image = Image.fromarray((scaled * 255).clip(0, 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.feature_extractor = None
        self.normal_center = None
        self.thresholds: dict[str, Any] = {}
        self.load_error: str | None = None
        self.ae_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        required = [
            ARTIFACT_DIR / "autoencoder_state.pt",
            ARTIFACT_DIR / "resnet_normal_center.npy",
            ARTIFACT_DIR / "thresholds.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            self.load_error = "Missing model artifacts: " + ", ".join(missing)
            return

        try:
            self.thresholds = json.loads((ARTIFACT_DIR / "thresholds.json").read_text(encoding="utf-8"))
            self.autoencoder = ConvAutoencoder().to(self.device)
            state = torch.load(ARTIFACT_DIR / "autoencoder_state.pt", map_location=self.device, weights_only=True)
            self.autoencoder.load_state_dict(state)
            self.autoencoder.eval()

            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            # Keep the convolutional feature map (B, 512, H, W); global-average-pooling
            # below reproduces the notebook embedding while retaining a coarse heatmap.
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)
            self.feature_extractor.eval()
            self.resnet_transform = weights.transforms()
            self.normal_center = np.load(ARTIFACT_DIR / "resnet_normal_center.npy").astype(np.float32).reshape(1, -1)
        except Exception as exc:
            self.load_error = f"Model loading failed: {type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.load_error is None and self.autoencoder is not None and self.feature_extractor is not None

    def _check_ready(self):
        if not self.ready:
            raise RuntimeError(self.load_error or "Models are not ready")

    @staticmethod
    def _open_image(image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width < 16 or image.height < 16:
                raise ValueError("Image is too small; upload an image at least 16 × 16 pixels.")
            if image.width * image.height > 25_000_000:
                raise ValueError("Image is too large; maximum supported size is 25 megapixels.")
            return image
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read image: {exc}") from exc

    def _autoencoder(self, image: Image.Image) -> dict[str, Any]:
        x = self.ae_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            reconstruction = self.autoencoder(x)
            score = torch.mean((x - reconstruction) ** 2).item()
            heatmap = torch.mean(torch.abs(x - reconstruction), dim=1)[0].cpu().numpy()
        threshold = float(self.thresholds["autoencoder_threshold"])
        return {
            "model": "cnn_autoencoder",
            "anomaly_score": float(score),
            "threshold": threshold,
            "prediction": "potential_defect" if score > threshold else "normal",
            "heatmap_png_base64": _png_base64(heatmap),
        }

    def _resnet(self, image: Image.Image) -> dict[str, Any]:
        x = self.resnet_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feature_map = self.feature_extractor(x)  # B x 512 x H x W
            embedding = feature_map.mean(dim=(2, 3)).cpu().numpy()
            score = float(np.linalg.norm(embedding - self.normal_center, axis=1)[0])

            # Coarse spatial feature-distance map, not pixel-level segmentation.
            fmap = feature_map[0].cpu().numpy()  # C x H x W
            center = self.normal_center.reshape(-1, 1, 1)
            heatmap = np.sqrt(np.mean((fmap - center) ** 2, axis=0))
        threshold = float(self.thresholds["resnet_threshold"])
        return {
            "model": "resnet18_feature_distance",
            "anomaly_score": score,
            "threshold": threshold,
            "prediction": "potential_defect" if score > threshold else "normal",
            "heatmap_png_base64": _png_base64(heatmap),
        }

    def predict(self, image_bytes: bytes, model_choice: str = "both") -> dict[str, Any]:
        self._check_ready()
        image = self._open_image(image_bytes)
        results = {}
        if model_choice in ("both", "autoencoder"):
            results["autoencoder"] = self._autoencoder(image)
        if model_choice in ("both", "resnet"):
            results["resnet"] = self._resnet(image)
        return {
            "image_width": image.width,
            "image_height": image.height,
            "results": results,
            "notice": "Anomaly scores are model-specific and are not probabilities. Heatmaps are visual indicators, not pixel-accurate defect masks.",
        }
