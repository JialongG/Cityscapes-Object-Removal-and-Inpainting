from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def replace_classifier_head(model: nn.Module, num_classes: int = 19) -> nn.Module:
    model.classifier = DeepLabHead(2048, num_classes)
    return model


def freeze_backbone_layers(model: nn.Module) -> Optional[nn.Module]:
    if not isinstance(model.backbone, nn.Module):
        return None
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


def segmentation_forward_pass(model: nn.Module, image_batch: torch.Tensor) -> torch.Tensor:
    return model(image_batch)["out"]


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one supervised DeepLab training epoch."""

    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = segmentation_forward_pass(model, images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate DeepLab loss on a validation dataloader."""

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = segmentation_forward_pass(model, images)
            total_loss += criterion(outputs, masks).item()

    return total_loss / len(dataloader)
