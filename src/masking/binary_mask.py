from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch


def build_class_binary_mask(mask: torch.Tensor, class_id: int) -> torch.Tensor:
    """Extract a single class mask and encode foreground as 255."""

    if mask.ndim == 3:
        # (B, H, W)
        return (mask == class_id).to(torch.uint8) * 255
    if mask.ndim == 2:
        return (mask == class_id).to(torch.uint8) * 255
    raise ValueError("mask must be (H, W) or (B, H, W)")


def combine_binary_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    """Merge multiple binary masks with pixel-wise union."""

    merged = None
    for mask in masks:
        current = (mask > 0).astype(np.uint8) * 255
        merged = current if merged is None else np.maximum(merged, current)
    if merged is None:
        raise ValueError("No masks were provided")
    return merged


def refine_binary_mask(mask: np.ndarray, kernel_size: int = 7, dilate_iter: int = 1, blur_size: int = 9) -> np.ndarray:
    """
    Expand and smooth mask boundaries (both for DeepLab and PatchGAN approaches).

    The dilation step tends to reduce edge misses from segmentation outputs,
    while blur softens transitions for inpainting models.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
    blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
    return cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)


def apply_mask_hole(image_tensor: torch.Tensor, binary_mask: torch.Tensor, fill_value: float = -1.0) -> torch.Tensor:
    """Apply a binary mask to an image batch by replacing masked pixels with fill_value."""

    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must be shaped (B, C, H, W)")
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.unsqueeze(1)
    if binary_mask.shape[0] != image_tensor.shape[0]:
        raise ValueError("Mask batch size must match image batch size")
    holed = image_tensor.clone()
    hole = binary_mask > 0
    holed[hole.expand_as(holed)] = fill_value
    return holed
