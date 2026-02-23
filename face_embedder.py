from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

PathLike = Union[str, Path]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class FaceEmbedder:
    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @torch.no_grad()
    def embed_one(self, img_path: PathLike) -> np.ndarray:
        img_path = Path(img_path)
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(self.device)
        e = self.model(x).cpu().numpy()[0].astype(np.float32)
        e /= (np.linalg.norm(e) + 1e-12)
        return e