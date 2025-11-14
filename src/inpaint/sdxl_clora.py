from __future__ import annotations

from typing import Optional

import torch
from PIL import Image


def load_sdxl_inpaint_pipeline(
    model_id: str,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
    variant: Optional[str] = None,
):
    from diffusers import AutoPipelineForInpainting

    device_type = torch.device(device).type
    if torch_dtype is None:
        torch_dtype = torch.float16 if device_type == "cuda" else torch.float32
    if variant is None and torch_dtype == torch.float16:
        variant = "fp16"

    kwargs = {"torch_dtype": torch_dtype}
    if variant is not None:
        kwargs["variant"] = variant
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    return pipe


def load_clora_weights(pipe, lora_path: str, adapter_name: Optional[str] = None, adapter_scale: float = 1.0):
    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    if adapter_name:
        pipe.set_adapters([adapter_name], adapter_weights=[adapter_scale])
    return pipe


def run_sdxl_inpaint(
    pipe,
    prompt: str,
    image: Image.Image,
    mask_image: Image.Image,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    strength: float = 0.99,
    seed: Optional[int] = None,
) -> Image.Image:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
    )
    return result.images[0]
