from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.inpaint.patchgan import load_generator_weights, validate_generator_input_size
from src.masking.binary_mask import refine_binary_mask
from src.segmentation.florence_sam import (
    apply_sam_refine_from_boxes,
    florence_refexp_segmentation,
    load_florence_model,
    load_sam_predictor,
    polygons_to_binary_mask,
    polygons_to_bboxes,
)


def run_baseline_pipeline(
    image_paths: Iterable[str],
    output_dir: str,
    patchgan_checkpoint: str,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    florence_model_id: str = "microsoft/Florence-2-large",
    prompt: str = "segment every car separately, output each car as an individual instance",
) -> List[Dict[str, str]]:
    """Run the control pipeline: Florence-2 (+ optional SAM refinement) -> binary mask -> PatchGAN."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor, florence_model = load_florence_model(model_id=florence_model_id, device=str(device))
    sam_predictor = load_sam_predictor(sam_checkpoint, model_type=sam_model_type, device=str(device)) if sam_checkpoint else None
    generator = load_generator_weights(checkpoint_path=patchgan_checkpoint, device=device)
    transform = T.Compose([T.ToTensor()])

    results: List[Dict[str, str]] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        validate_generator_input_size(height=image.height, width=image.width)

        florence_out = florence_refexp_segmentation(image, processor, florence_model, text_input=prompt)
        polygons = florence_out.get("polygons", [])

        if sam_predictor is not None:
            # Use SAM to refine Florence detections into sharper per-instance masks.
            bboxes = polygons_to_bboxes(polygons)
            if bboxes.shape[0] == 0:
                binary_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                binary_mask = apply_sam_refine_from_boxes(np.array(image), sam_predictor, bboxes)
        else:
            binary_mask = polygons_to_binary_mask(polygons, (image.width, image.height))

        refined_mask = refine_binary_mask(binary_mask)

        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tanh = image_tensor * 2 - 1
        mask_tensor = torch.from_numpy((refined_mask > 0).astype(np.uint8)).unsqueeze(0).to(device)
        holed = image_tanh.clone()

        # In GAN branch, -1 corresponds to black in tanh space:
        holed[mask_tensor.unsqueeze(1).expand_as(holed) > 0] = -1
        with torch.no_grad():
            generated_tanh = generator(holed)
        generated = ((generated_tanh.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        generated_img = Image.fromarray((generated * 255).astype(np.uint8))

        stem = Path(image_path).stem
        mask_file = output_path / f"{stem}_mask.png"
        result_file = output_path / f"{stem}_patchgan.png"
        Image.fromarray(refined_mask).save(mask_file)
        generated_img.save(result_file)
        results.append({"image": image_path, "mask": str(mask_file), "result": str(result_file)})

    return results
