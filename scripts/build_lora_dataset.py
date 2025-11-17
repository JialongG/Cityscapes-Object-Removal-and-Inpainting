import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lora.caption_pack import build_caption_dataset
from src.lora.dataset_filter import collect_no_class_images
from src.utils.config import load_yaml, require_existing_dir, require_existing_file, require_keys


def main():
    parser = argparse.ArgumentParser(description="Build CLoRA training dataset from Cityscapes")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    require_keys(
        cfg,
        ["cityscapes_leftimg_root", "seg_checkpoint", "filtered_output_dir", "lora_dataset_dir"],
        args.config,
    )
    require_existing_dir(cfg["cityscapes_leftimg_root"], "cityscapes_leftimg_root")
    require_existing_file(cfg["seg_checkpoint"], "seg_checkpoint")

    saved = collect_no_class_images(
        cityscapes_leftimg_root=cfg["cityscapes_leftimg_root"],
        seg_checkpoint=cfg["seg_checkpoint"],
        output_dir=cfg["filtered_output_dir"],
        class_id=cfg.get("class_id", 13),
        max_images=cfg.get("max_images", 50),
    )
    packed = build_caption_dataset(
        source_dir=cfg["filtered_output_dir"],
        output_dir=cfg["lora_dataset_dir"],
        caption_text=cfg.get(
            "caption_text",
            "clean street background, no vehicles, empty road, realistic city environment",
        ),
    )
    print(f"Filtered images: {saved}")
    print(f"Packed pairs: {packed}")


if __name__ == "__main__":
    main()
