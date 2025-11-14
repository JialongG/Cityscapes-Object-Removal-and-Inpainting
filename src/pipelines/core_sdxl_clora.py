from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.inpaint.sdxl_clora import load_clora_weights, load_sdxl_inpaint_pipeline, run_sdxl_inpaint
from src.masking.binary_mask import build_class_binary_mask, refine_binary_mask
from src.segmentation.deeplab import segmentation_forward_pass


def load_deeplab_checkpoint(checkpoint_path: str, device: torch.device):
    # Load the full trained module:
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def infer_deeplab_class_mask(image: Image.Image, seg_model, class_id: int, device: torch.device) -> np.ndarray:
    # DeepLab was trained with ImageNet normalization, and thus inference must match it:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = segmentation_forward_pass(seg_model, inp)
        pred_mask = torch.argmax(logits.squeeze(0), dim=0)
    # Keep mask representation explicit as uint8 in [0, 255]:
    binary = build_class_binary_mask(pred_mask, class_id=class_id).cpu().numpy()
    return binary.astype(np.uint8)


def run_core_pipeline(
    image_paths: Iterable[str],
    output_dir: str,
    seg_checkpoint: str,
    sdxl_model_id: str,
    lora_path: Optional[str],
    class_id: int = 13,
    prompt: str = "clean realistic city street background, no vehicles",
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    steps: int = 30,
    strength: float = 0.99,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Run the main pipeline: DeepLab -> binary mask -> SDXL+CLoRA."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seg_model = load_deeplab_checkpoint(seg_checkpoint, device=device)
    pipe = load_sdxl_inpaint_pipeline(sdxl_model_id, device=str(device))
    if lora_path:
        load_clora_weights(pipe, lora_path=lora_path)

    results: List[Dict[str, str]] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        # Segmentation mask drives the editable region for inpainting:
        binary_mask = infer_deeplab_class_mask(image, seg_model, class_id=class_id, device=device)
        refined_mask = refine_binary_mask(binary_mask)
        mask_pil = Image.fromarray(refined_mask, mode="L")
        generated = run_sdxl_inpaint(
            pipe=pipe,
            prompt=prompt,
            image=image,
            mask_image=mask_pil,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            strength=strength,
            seed=seed,
        )

        stem = Path(image_path).stem
        mask_file = output_path / f"{stem}_mask.png"
        result_file = output_path / f"{stem}_sdxl_clora.png"
        mask_pil.save(mask_file)
        generated.save(result_file)
        results.append({"image": image_path, "mask": str(mask_file), "result": str(result_file)})

    return results

