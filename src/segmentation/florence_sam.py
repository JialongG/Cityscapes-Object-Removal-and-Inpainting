from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class FlorenceSamComponents:
    processor: Any
    florence_model: Any
    sam_predictor: Any


def load_florence_model(model_id: str = "microsoft/Florence-2-large", device: str = "cuda"):
    """Load Florence-2 model and processor."""

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def load_sam_predictor(checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"):
    """Build a SAM predictor from a `segment_anything` checkpoint."""

    from segment_anything import SamPredictor, sam_model_registry

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    return SamPredictor(sam_model)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_flat_polygon(candidate: Any) -> bool:
    return (
        isinstance(candidate, Sequence)
        and len(candidate) >= 6
        and all(_is_number(value) for value in candidate)
    )


def iter_flat_polygons(polygons: Iterable[Any]) -> Iterator[List[float]]:
    """
    Yield flat ``[x0, y0, x1, y1, ...]`` polygons from Florence output.

    Florence referring-expression segmentation returns polygons grouped by
    instance, e.g. ``[[[x0, y0, ...]], [[x0, y0, ...]]]``. The helper also
    accepts already-flat polygon lists so callers can work with either shape.
    """

    if _is_flat_polygon(polygons):
        yield [float(value) for value in polygons]
        return

    for item in polygons:
        if _is_flat_polygon(item):
            yield [float(value) for value in item]
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from iter_flat_polygons(item)


def polygons_to_bboxes(polygons: Iterable[Any]) -> np.ndarray:
    """Derive axis-aligned bounding boxes (xyxy) from flat polygon coordinates."""

    boxes: List[List[float]] = []
    for polygon in iter_flat_polygons(polygons):
        xs = polygon[0::2]
        ys = polygon[1::2]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def polygons_to_binary_mask(polygons: Iterable[Any], image_size: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon coordinates into a binary mask in [0, 255]."""

    width, height = image_size
    mask = Image.new("L", (width, height), color=0)
    drawer = ImageDraw.Draw(mask)
    for polygon in iter_flat_polygons(polygons):
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        drawer.polygon(points, fill=255)
    return np.array(mask, dtype=np.uint8)


def florence_refexp_segmentation(
    image: Image.Image,
    processor: Any,
    model: Any,
    text_input: str = "segment every car separately",
    task_prompt: str = "<REFERRING_EXPRESSION_SEGMENTATION>",
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run Florence referring-expression segmentation and return parsed task payload."""

    inputs = processor(text=task_prompt + text_input, images=image, return_tensors="pt")
    for k in inputs:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )

    # Keep return shape stable even when the task key is missing:
    return parsed.get(task_prompt, {})


def apply_sam_refine_from_boxes(image_np: np.ndarray, predictor: Any, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Refine coarse detections with SAM and merge all instance masks."""

    predictor.set_image(image_np)
    masks = []
    for box in boxes_xyxy:
        mask_batch, _, _ = predictor.predict(box=box, multimask_output=False)
        masks.append(mask_batch[0].astype(np.uint8))
    if not masks:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)
    merged = np.clip(np.sum(np.stack(masks, axis=0), axis=0), 0, 1)
    return (merged * 255).astype(np.uint8)
