# Cityscapes Object Removal and Inpainting

This repository contains two object-removal pipelines for Cityscapes-style street-view images:

- Main pipeline: **DeepLab -> binary mask -> SDXL + CLoRA**
- Control pipeline: **Florence-2 (+ optional SAM refinement) -> binary mask -> PatchGAN**

The migrated implementation lives in `src/`, with command-line entry points in `scripts/` and runtime parameters in `configs/`. The original notebooks remain in the repository as reference implementations for comparing migration behavior and experiment history.

## Project Scope

The project combines semantic segmentation and inpainting for selective removal of foreground objects such as vehicles and pedestrians from street scenes.

Task 1, image segmentation:

- Predict pixel-level class labels for a Cityscapes image.
- Produce semantic masks with Cityscapes train IDs.
- Use DeepLabv3-ResNet50 for the main path.
- Use Florence-2 with optional SAM refinement for the control mask path.

Task 2, image inpainting:

- Convert the target class prediction to a binary removal mask.
- Inpaint the masked region with either SDXL + CLoRA or PatchGAN.
- Save both the binary mask and the inpainted image.

## Dataset

Cityscapes fine annotations contain 5,000 images across train, validation, and test splits. The code expects the standard directory layout:

```text
leftImg8bit/<split>/<city>/*_leftImg8bit.png
gtFine/<split>/<city>/*_gtFine_labelIds.png
```

The data loader maps Cityscapes label IDs to the 19 train IDs used for evaluation. Train ID `255` is treated as ignore index.

Cityscapes Image Example

## Repository Layout

```text
src/
  data.py                         Cityscapes dataset and tensor utilities
  segmentation/                   DeepLab and Florence/SAM wrappers
  masking/                        binary mask creation and refinement
  inpaint/                        SDXL+CLoRA and PatchGAN components
  pipelines/                      end-to-end pipeline orchestration
  lora/                           CLoRA dataset filtering and caption files
  utils/                          config loading and validation
scripts/                          CLI entry points
configs/                          YAML runtime configuration templates
project/                          compatibility re-exports for older imports
assets/                           README images
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts add the repository root to `sys.path`, so they can be run from a fresh checkout without installing the package in editable mode.

## External Assets

The following files are not versioned in git and must be supplied locally:

- DeepLab checkpoint, for example `checkpoints/deeplab_cityscapes_full_model_3.pth`
- PatchGAN generator checkpoint, for example `checkpoints/gan_generator_head_3.pth`
- CLoRA adapter weights, for example `checkpoints/lora.safetensors`
- Optional SAM checkpoint for control-pipeline mask refinement
- Hugging Face access for the SDXL inpainting model

The sample configs under `configs/` use local placeholder paths. Update them before running the scripts.

## Run Pipelines

Main pipeline:

```bash
python scripts/run_core_pipeline.py --config configs/core_sdxl_clora.yaml
```

Control pipeline:

```bash
python scripts/run_baseline_pipeline.py --config configs/baseline_florence_patchgan.yaml
```

Build a CLoRA caption dataset:

```bash
python scripts/build_lora_dataset.py --config configs/lora_dataset.yaml
```

PatchGAN uses a six-level U-Net generator. Input images for that branch must have height and width divisible by 64; Cityscapes native resolution and the project crop size satisfy this requirement.

## CLoRA Training Data

The CLoRA branch uses a curated set of street scenes without the target class. `scripts/build_lora_dataset.py` filters Cityscapes images with the DeepLab checkpoint, then writes image and caption pairs for OneTrainer.

LoRA training itself was performed with OneTrainer rather than a custom training script.

OneTrainerGUI

## Results

Mask examples:

Mask Prediction Result 1
Mask Prediction Result 2

Inpainting example:

Final In-Painting Result 1

## Acknowledgements

This project builds on open-source models, datasets, and tooling from the community:

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion XL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) for LoRA workflows
- [OneTrainer](https://github.com/Nerogar/OneTrainer) for LoRA training