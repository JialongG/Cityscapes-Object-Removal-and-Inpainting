from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.segmentation.deeplab import segmentation_forward_pass


def collect_no_class_images(
    cityscapes_leftimg_root: str,
    seg_checkpoint: str,
    output_dir: str,
    class_id: int = 13,
    max_images: int = 50,
    splits: Iterable[str] = ("train", "val"),
) -> int:
    """
    Build a no-target-class image subset for LoRA training data preparation.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(seg_checkpoint, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for split in splits:
        split_dir = Path(cityscapes_leftimg_root) / split
        if not split_dir.exists():
            continue
        for city_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                image = Image.open(image_path).convert("RGB")
                inp = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = segmentation_forward_pass(model, inp)
                    pred_mask = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()
                if np.any(pred_mask == class_id):
                    
                    # Skip images where the target class still appears.
                    continue
                image.save(out / image_path.name)
                saved += 1
                if saved >= max_images:
                    return saved
    return saved

