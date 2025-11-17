from __future__ import annotations

import shutil
from pathlib import Path


def build_caption_dataset(
    source_dir: str,
    output_dir: str,
    caption_text: str = "clean street background, no vehicles, empty road, realistic city environment",
) -> int:
    src = Path(source_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg")))
    for idx, img_path in enumerate(image_paths):
        new_name = f"{idx:05d}.png"
        target_img = dst / new_name
        shutil.copy(img_path, target_img)
        (dst / f"{idx:05d}.txt").write_text(caption_text, encoding="utf-8")
    return len(image_paths)

