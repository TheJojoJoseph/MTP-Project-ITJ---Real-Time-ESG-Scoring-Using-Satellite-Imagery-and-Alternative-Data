from typing import Dict, List, Tuple

import torch
from torch import Tensor
from PIL import Image
from transformers import ViTModel, ViTImageProcessor


class ViTFeatureExtractor:
    """Wrap a pre-trained ViT model for satellite patch embeddings.

    Uses Hugging Face transformers (e.g., "google/vit-base-patch16-224").
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str | None = None) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_batch(self, images: List[Image.Image]) -> Tensor:
        """Return CLS embeddings for a batch of images.

        Parameters
        ----------
        images : list of PIL.Image
            Images should already be RGB; resizing/normalization is
            handled by the processor.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, D)
        return cls_embeddings.cpu()


def encode_company_images(
    image_map: Dict[str, Image.Image],
    extractor: ViTFeatureExtractor,
) -> Dict[str, Tensor]:
    """Encode all company images into a dict of CLS embeddings."""
    company_ids = list(image_map.keys())
    images = [image_map[cid] for cid in company_ids]
    feats = extractor.encode_batch(images)
    return {cid: feats[i] for i, cid in enumerate(company_ids)}


def get_dummy_attention_map(image: Image.Image) -> Tensor:
    """Return a dummy attention map for visualization.

    True multi-head attention visualization requires hooks into the
    internal attention tensors; to keep this notebook lightweight and
    backend-agnostic, we provide a simple Gaussian-like heatmap as a
    placeholder.
    """
    import numpy as np

    size = 14  # corresponds roughly to 16x16 patches on 224x224
    x = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, x)
    sigma = 0.5
    heat = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    heat = heat / heat.max()
    return torch.from_numpy(heat).float()
