"""
Fast data loader helpers for ImageNet.

Provides:
- fast_image_loader(path) -> HWC uint8 numpy array using torchvision.io.read_image
- create_dataloader(dataset, config, shuffle) -> torch.utils.data.DataLoader with prefetch_factor and safety defaults
- PrefetchLoader to asynchronously move batches to device (non-blocking)

Usage:
    from core.fast_loader import fast_image_loader, create_dataloader, PrefetchLoader

    train_dataset = datasets.ImageFolder(train_dir, transform=..., loader=fast_image_loader)
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    prefetch_loader = PrefetchLoader(train_loader, device)

    for images, targets in prefetch_loader:
        # images and targets are on device already (if device is cuda)
        ...

Author: generated
Date: 2025-10-29
"""

import os
from typing import Any, Dict, Iterator, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.io as tv_io


def fast_image_loader(path: str) -> np.ndarray:
    """
    Fast image loader using torchvision.io.read_image.

    Returns HWC uint8 numpy array which is compatible with Albumentations.
    Falls back to PIL if torchvision fails for any reason.
    """
    try:
        # tv_io.read_image -> CHW, uint8
        img = tv_io.read_image(path)
        if not isinstance(img, torch.Tensor):
            raise RuntimeError("read_image did not return a Tensor")
        # Convert to HWC numpy array
        img = img.permute(1, 2, 0).contiguous().numpy()
        return img
    except Exception:
        # Fallback to PIL (slower) to keep robustness
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
        return arr


def _move_to_device(item: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested structures to device (non_blocking=True)."""
    if torch.is_tensor(item):
        return item.to(device, non_blocking=True)
    if isinstance(item, dict):
        return {k: _move_to_device(v, device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        seq = [_move_to_device(x, device) for x in item]
        return type(item)(seq)
    return item


class PrefetchLoader:
    """
    Wraps a DataLoader and moves batches to `device` before yielding.

    If device is CUDA, a CUDA stream is used for asynchronous copies.

    Example:
        loader = create_dataloader(...)
        prefetch = PrefetchLoader(loader, torch.device('cuda'))
        for images, targets in prefetch:
            # images, targets are on GPU
            train_step(images, targets)
    """

    def __init__(self, loader: Iterable, device: torch.device):
        self.loader = loader
        self.device = device
        self.cuda = (device.type == "cuda")
        self.stream = torch.cuda.Stream(device=device) if self.cuda else None

    def __iter__(self) -> Iterator:
        if not self.cuda:
            # CPU device -> simply move synchronously
            for batch in self.loader:
                yield _move_to_device(batch, self.device)
            return

        # CUDA path: attempt to overlap host->device copies using a dedicated stream
        for batch in self.loader:
            # Move to device on the prefetch stream
            with torch.cuda.stream(self.stream):
                batch_on_device = _move_to_device(batch, self.device)
            # Wait for the copy to finish on the default stream before user code uses it
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            yield batch_on_device

    def __len__(self) -> int:
        try:
            return len(self.loader)
        except Exception:
            raise TypeError("Underlying loader has no length")


def create_dataloader(dataset: Any, config: Any, shuffle: bool = False) -> DataLoader:
    """
    Create a DataLoader with recommended performance settings pulled from `config`.

    Expected config attributes (will use safe defaults if missing):
      - batch_size
      - num_workers
      - pin_memory
      - persistent_workers
      - prefetch_factor (optional)

    Returns a torch.utils.data.DataLoader ready to use.
    """
    batch_size = getattr(config, "batch_size", 32)
    num_workers = getattr(config, "num_workers", 8)
    pin_memory = getattr(config, "pin_memory", True)
    persistent_workers = getattr(config, "persistent_workers", True) if num_workers > 0 else False
    prefetch_factor = getattr(config, "prefetch_factor", 4)

    # Defensive sanity: ensure prefetch_factor is integer >=1
    try:
        pf = int(prefetch_factor)
        if pf < 1:
            pf = 2
    except Exception:
        pf = 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=pf,
    )


# Utility to suggest reasonable num_workers based on available CPUs
def suggested_num_workers(limit: Optional[int] = None) -> int:
    """Return a safe default num_workers (min(available_cpus//2, limit))."""
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    suggestion = max(1, cpus // 2)
    if limit:
        suggestion = min(suggestion, limit)
    return suggestion


# If invoked directly, demonstrate basic behavior (does not run heavy operations)
if __name__ == "__main__":
    print("fast_loader module: simple sanity check")
    print("Suggested num_workers:", suggested_num_workers())
