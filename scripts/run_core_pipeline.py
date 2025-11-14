import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.core_sdxl_clora import run_core_pipeline
from src.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Run DeepLab -> binary mask -> SDXL+CLoRA pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    image_paths = sorted(str(p) for p in Path(cfg["input_dir"]).glob(cfg.get("glob", "*.png")))

    results = run_core_pipeline(
        image_paths=image_paths,
        output_dir=cfg["output_dir"],
        seg_checkpoint=cfg["seg_checkpoint"],
        sdxl_model_id=cfg["sdxl_model_id"],
        lora_path=cfg.get("lora_path"),
        class_id=cfg.get("class_id", 13),
        prompt=cfg.get("prompt", "clean realistic city street background, no vehicles"),
        negative_prompt=cfg.get("negative_prompt"),
        guidance_scale=cfg.get("guidance_scale", 7.5),
        steps=cfg.get("steps", 30),
        strength=cfg.get("strength", 0.99),
        seed=cfg.get("seed"),
    )
    print(f"Finished {len(results)} images.")


if __name__ == "__main__":
    main()
