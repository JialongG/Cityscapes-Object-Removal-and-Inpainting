from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.segmentation.deeplab import segmentation_forward_pass

PATCHGAN_SIZE_MULTIPLE = 64


def create_holed_image(image_batch: torch.Tensor, mask_batch: torch.Tensor, class_id: int) -> torch.Tensor:
    """Zero-out a target class region in a normalized image batch."""

    holed_image = image_batch.clone()
    binary_mask = (mask_batch == class_id).unsqueeze(1)
    holed_image[binary_mask.expand_as(holed_image)] = -1
    return holed_image


def validate_generator_input_size(height: int, width: int, multiple: int = PATCHGAN_SIZE_MULTIPLE) -> None:
    """Validate spatial dimensions for the six-level U-Net generator."""

    if height < multiple or width < multiple:
        raise ValueError(f"PatchGAN input must be at least {multiple}x{multiple}; got {height}x{width}.")
    if height % multiple != 0 or width % multiple != 0:
        raise ValueError(
            f"PatchGAN input height and width must be divisible by {multiple}; got {height}x{width}."
        )


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class Generator(nn.Module):
    """U-Net style generator used by the PatchGAN branch."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)


class Discriminator(nn.Module):
    """Patch discriminator that scores local realism."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def block(in_filters: int, out_filters: int, normalize: bool = True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


def build_gan_models() -> Tuple[Generator, Discriminator]:
    return Generator(), Discriminator()


def load_generator_weights(checkpoint_path: str, device: torch.device) -> Generator:
    generator = Generator().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def train_gan_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real_images: torch.Tensor,
    segmentation_masks: torch.Tensor,
    class_to_remove: int = 13,
    adv_loss_fn: Optional[nn.Module] = None,
    l1_loss_fn: Optional[nn.Module] = None,
    lambda_l1: float = 100.0,
    g_optimizer: Optional[torch.optim.Optimizer] = None,
    d_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run a single adversarial optimization step.

    Inputs are expected to be in tanh range [-1, 1].
    """

    if adv_loss_fn is None:
        adv_loss_fn = nn.BCEWithLogitsLoss()
    if l1_loss_fn is None:
        l1_loss_fn = nn.L1Loss()

    device = real_images.device
    holed_images = create_holed_image(real_images, segmentation_masks, class_to_remove)
    fake_images = generator(holed_images)
    pred_real = discriminator(real_images)
    pred_fake_detached = discriminator(fake_images.detach())
    labels_real = torch.ones_like(pred_real, device=device)
    labels_fake = torch.zeros_like(pred_fake_detached, device=device)
    d_loss = 0.5 * (adv_loss_fn(pred_real, labels_real) + adv_loss_fn(pred_fake_detached, labels_fake))

    if d_optimizer is not None:
        d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        d_optimizer.step()

    pred_fake = discriminator(fake_images)
    labels_gen = torch.ones_like(pred_fake, device=device)
    g_loss_adv = adv_loss_fn(pred_fake, labels_gen)

    # L1 keeps the generated result close to source context outside masked area:
    g_loss_l1 = l1_loss_fn(fake_images, real_images)
    g_loss = g_loss_adv + (lambda_l1 * g_loss_l1)

    if g_optimizer is not None:
        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        g_optimizer.step()

    return {
        "g_loss": g_loss.detach(),
        "d_loss": d_loss.detach(),
        "g_loss_adv": g_loss_adv.detach(),
        "g_loss_l1": g_loss_l1.detach(),
        "fake_images": fake_images.detach(),
        "holed_images": holed_images.detach(),
    }


def train_gan_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    seg_model: nn.Module,
    class_to_remove: int,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    adv_loss: nn.Module,
    l1_loss: nn.Module,
    lambda_l1: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one PatchGAN training epoch using DeepLab masks as hole locations."""

    generator.train()
    discriminator.train()
    seg_model.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for real_images_imagenet, _ in tqdm(dataloader, desc="GAN Training"):
        real_images_imagenet = real_images_imagenet.to(device)

        with torch.no_grad():
            seg_logits = segmentation_forward_pass(seg_model, real_images_imagenet)
            pred_masks = torch.argmax(seg_logits, dim=1)

        real_images_01 = (real_images_imagenet * std + mean).clamp(0, 1)
        real_images_gan = real_images_01 * 2 - 1
        losses = train_gan_step(
            generator=generator,
            discriminator=discriminator,
            real_images=real_images_gan,
            segmentation_masks=pred_masks,
            class_to_remove=class_to_remove,
            adv_loss_fn=adv_loss,
            l1_loss_fn=l1_loss,
            lambda_l1=lambda_l1,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
        )
        total_g_loss += float(losses["g_loss"].item())
        total_d_loss += float(losses["d_loss"].item())

    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)
