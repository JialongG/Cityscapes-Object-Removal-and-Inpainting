"""
Core modules for object-removal pipelines.

Main pipeline:
    DeepLab -> binary mask -> SDXL + CLoRA

Control pipeline:
    Florence-2 + SAM -> binary mask -> PatchGAN
"""
