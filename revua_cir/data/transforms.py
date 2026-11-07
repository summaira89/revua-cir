"""Image preprocessing. Uses CLIP's standard normalization constants."""
from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class _MissingImageTally:
    """Counts gray-placeholder substitutions so a misconfigured data root cannot
    pass silently. The fallback below is deliberate (FashionIQ/CIRR link rot,
    Section 9) but it is indistinguishable from a wrong `data.root`: with a bad
    path *every* image becomes gray and training runs to completion on noise.
    Datasets call `report()` after construction to surface the rate."""

    def __init__(self):
        self.missing = 0
        self.total = 0
        self.examples: list[str] = []

    def hit(self, path: str):
        self.missing += 1
        if len(self.examples) < 5:
            self.examples.append(path)

    def snapshot(self):
        return self.missing, self.total, list(self.examples)


MISSING_IMAGES = _MissingImageTally()


def safe_load_transform(path: str, transform, fallback_size: int = 224) -> torch.Tensor:
    """Loads + transforms an image; on failure (broken/missing file -- expected
    given FashionIQ/CIRR link-rot risk flagged in the paper framework, Section 9)
    returns a mid-gray placeholder tensor instead of crashing the whole batch.

    Substitutions are tallied in `MISSING_IMAGES`; check it before trusting a run
    (see `verify_data.py`), because a wrong data root looks exactly like total
    link rot."""
    MISSING_IMAGES.total += 1
    try:
        img = load_image(path)
        return transform(img)
    except (FileNotFoundError, OSError):
        MISSING_IMAGES.hit(path)
        placeholder = Image.new("RGB", (fallback_size, fallback_size), (127, 127, 127))
        return transform(placeholder)
