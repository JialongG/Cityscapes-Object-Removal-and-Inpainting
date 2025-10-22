import os
from glob import glob
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class CityscapesDataset(Dataset):
    TRAIN_ID_TO_COLOR = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32],
    ]

    LABEL_ID_TO_TRAIN_ID = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1,
        9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
        17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
        -1: 255,
    }

    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.mask_dir = os.path.join(root_dir, "gtFine", split)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*_leftImg8bit.png")))
        self.mask_paths = sorted(glob(os.path.join(self.mask_dir, "*", "*_gtFine_labelIds.png")))
        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError("Mismatched image and mask counts.")
        if not self.image_paths:
            raise RuntimeError(f"No Cityscapes images found at {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_pil = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx])
        image_tensor = TF.to_tensor(image_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long().unsqueeze(0)
        if self.transforms:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)
        mask_tensor = mask_tensor.squeeze(0)
        train_id_mask = torch.full_like(mask_tensor, 255, dtype=torch.long)
        for label_id, train_id in self.LABEL_ID_TO_TRAIN_ID.items():
            train_id_mask[mask_tensor == label_id] = train_id
        return image_tensor, train_id_mask


PairedTransform = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


def create_paired_random_crop(size: Tuple[int, int]) -> PairedTransform:
    def do_crop(image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        i, j, h, w = T.RandomCrop.get_params(image, output_size=size)
        return TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)
    return do_crop


def create_paired_random_flip(p: float = 0.5) -> PairedTransform:
    def do_flip(image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask
    return do_flip


def create_paired_center_crop(size: Tuple[int, int]) -> PairedTransform:
    def do_center_crop(image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(image, list(size)), TF.center_crop(mask, list(size))
    return do_center_crop


def get_transforms(split: str, crop_size: Tuple[int, int] = (512, 1024)):
    normalize = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    spatial_fns: List[PairedTransform] = []
    if split == "train":
        spatial_fns.extend([create_paired_random_flip(0.5), create_paired_random_crop(crop_size)])
    else:
        spatial_fns.append(create_paired_center_crop(crop_size))

    def final_transform(image: Tensor, mask: Tensor):
        for fn in spatial_fns:
            image, mask = fn(image, mask)
        return normalize(image), mask

    return final_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    crop_size: Tuple[int, int] = (512, 1024),
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CityscapesDataset(data_dir, "train", get_transforms("train", crop_size))
    val_dataset = CityscapesDataset(data_dir, "val", get_transforms("val", crop_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_id_to_color(mask_tensor: torch.Tensor) -> np.ndarray:
    mask_np = mask_tensor.detach().cpu().numpy()
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for train_id, color in enumerate(CityscapesDataset.TRAIN_ID_TO_COLOR):
        color_mask[mask_np == train_id] = color
    return color_mask


def denormalize_imagenet(tensor_img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_img.device).view(1, 3, 1, 1)
    return ((tensor_img * std) + mean).clamp(0, 1)


def denormalize_tanh(tensor_img: torch.Tensor) -> torch.Tensor:
    return (tensor_img + 1) / 2
