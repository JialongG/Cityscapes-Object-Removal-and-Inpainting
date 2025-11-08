import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.baseline_florence_patchgan import run_baseline_pipeline
from src.utils.config import collect_input_files, load_yaml, require_existing_file, require_keys


def main():
    parser = argparse.ArgumentParser(description="Run Florence-2+SAM -> binary mask -> PatchGAN pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    require_keys(cfg, ["input_dir", "output_dir", "patchgan_checkpoint"], args.config)
    image_paths = collect_input_files(cfg["input_dir"], cfg.get("glob", "*.png"))
    require_existing_file(cfg["patchgan_checkpoint"], "patchgan_checkpoint")
    if cfg.get("sam_checkpoint"):
        require_existing_file(cfg["sam_checkpoint"], "sam_checkpoint")

    results = run_baseline_pipeline(
        image_paths=image_paths,
        output_dir=cfg["output_dir"],
        patchgan_checkpoint=cfg["patchgan_checkpoint"],
        sam_checkpoint=cfg.get("sam_checkpoint"),
        sam_model_type=cfg.get("sam_model_type", "vit_h"),
        florence_model_id=cfg.get("florence_model_id", "microsoft/Florence-2-large"),
        prompt=cfg.get("prompt", "segment every car separately, output each car as an individual instance"),
    )
    print(f"Finished {len(results)} images.")


if __name__ == "__main__":
    main()
