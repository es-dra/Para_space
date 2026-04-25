"""
Data loading utilities for INR experiments.

Provides dataset wrappers for standard image super-resolution benchmarks
(Set5, Set14, BSD100, DIV2K, etc.) and utilities for generating
coordinate grids and transformed image families.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset registry
DATASET_REGISTRY = {
    "Set5": {"hr_dir": "Set5/HR", "num_images": 5},
    "Set14": {"hr_dir": "Set14/HR", "num_images": 14},
    "Urban100": {"hr_dir": "Urban100/HR", "num_images": 100},
    "BSD100": {"hr_dir": "BSD100/HR", "num_images": 100},
    "DIV2K_train": {"hr_dir": "DIV2K_train_HR", "num_images": 800},
    "DIV2K_valid": {"hr_dir": "DIV2K_valid_HR", "num_images": 100},
}


class ImageDataset(Dataset):
    """
    Dataset wrapper for image files.

    Handles various dataset structures (Set5, DIV2K, BSD100, etc.)

    Args:
        data_root: Root directory containing dataset folders
        dataset_name: One of 'Set5', 'Set14', 'Urban100', 'BSD100', 'DIV2K_train', 'DIV2K_valid'
        split: Dataset split (for DIV2K, 'train' or 'valid')
        image_size: If specified, resize all images to this size
        transform: Optional transform to apply
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        split: str = "test",
        image_size: Optional[int] = None,
        transform: Optional[callable] = None,
    ):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # Determine the actual directory path
        if dataset_name in DATASET_REGISTRY:
            self.hr_dir = self.data_root / DATASET_REGISTRY[dataset_name]["hr_dir"]
        elif dataset_name == "DIV2K":
            # Handle DIV2K with split
            div2k_dir = self.data_root / "DIV2K"
            if split == "train":
                self.hr_dir = div2k_dir / "DIV2K_train_HR"
            else:
                self.hr_dir = div2k_dir / "DIV2K_valid_HR"
        else:
            # Try direct path
            self.hr_dir = self.data_root / dataset_name / "HR"

        # Collect image paths
        self.image_paths = sorted(self._collect_images())

    def _collect_images(self) -> List[Path]:
        """Collect all image paths from the dataset directory."""
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        images = []

        if self.hr_dir.exists():
            for ext in extensions:
                images.extend(self.hr_dir.glob(f"*{ext}"))
                images.extend(self.hr_dir.glob(f"*{ext.upper()}"))

        return sorted(images)

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        """
        Get image as tensor and metadata.

        Args:
            idx: Image index

        Returns:
            image: Shape (C, H, W) normalized to [0, 1]
            image_name: Original filename
            original_size: (H, W) before any resizing
        """
        img_path = self.image_paths[idx]
        image_name = img_path.name

        # Load image
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = (img.shape[0], img.shape[1])

        # Resize if specified
        if self.image_size is not None:
            img = cv2.resize(
                img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )

        # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
        img_tensor = (
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        )

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, image_name, original_size


def get_image_coordinates(
    height: int,
    width: int,
    pixel_spacing: float = 1.0,
    normalize: str = "center",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate coordinate grid for an image.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        pixel_spacing: Spacing between pixels (for physical coordinates)
        normalize: Coordinate normalization scheme:
            - 'center': Coordinates in [-1, 1] with center at origin
            - 'corner': Coordinates in [0, W] x [0, H]
            - 'none': Physical coordinates
        device: torch device

    Returns:
        coords: Shape (H*W, 2) or (H, W, 2) depending on flatten
    """
    if normalize == "center":
        # Create coordinates in [-1, 1] range with center at origin
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)
    elif normalize == "corner":
        # Create coordinates in [0, W] x [0, H]
        x = torch.arange(width, device=device, dtype=torch.float32)
        y = torch.arange(height, device=device, dtype=torch.float32)
    else:
        # Physical coordinates
        x = torch.arange(width, device=device, dtype=torch.float32) * pixel_spacing
        y = torch.arange(height, device=device, dtype=torch.float32) * pixel_spacing

    # Create meshgrid
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Stack to get (H, W, 2)
    coords = torch.stack([xx, yy], dim=-1)

    return coords


def image_to_tensor(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Convert numpy image (H, W, C) to torch tensor (C, H, W).

    Args:
        image: Numpy array of shape (H, W, C) or (H, W)
        normalize: Whether to normalize to [0, 1]. If image is already float in [0, 1], set to False.

    Returns:
        torch.Tensor of shape (C, H, W) or (1, H, W)
    """
    if image.ndim == 2:
        # Grayscale: (H, W) -> (1, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0)
    else:
        # Color: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1)

    if normalize:
        # Only normalize if image is in [0, 255] range (uint8)
        # If image is already float in [0, 1], don't normalize again
        if tensor.max() > 1.0:
            tensor = tensor.float() / 255.0

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor (C, H, W) or (B, C, H, W) to numpy image.

    Args:
        tensor: torch tensor

    Returns:
        numpy array of shape (H, W, C) or (H, W)
    """
    if tensor.ndim == 4:
        # (B, C, H, W) -> take first batch
        tensor = tensor[0]

    if tensor.ndim == 3:
        # (C, H, W) -> (H, W, C)
        if tensor.shape[0] in [1, 3]:
            array = tensor.permute(1, 2, 0).detach().cpu().numpy()
        else:
            array = tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        array = tensor.detach().cpu().numpy()

    # Denormalize if needed
    if array.max() <= 1.0:
        array = (array * 255).clip(0, 255).astype(np.uint8)

    return array


class TransformedImageFamily:
    """
    Generate a family of geometrically transformed images from a single image.

    Used for P3 experiment to generate {T_g · I} for g in parameter range.

    Args:
        base_image: Base image to transform (numpy array H, W, C or torch tensor)
        transform_type: Type of transformation ('rotation', 'scale', 'translation')
        param_range: Array of transform parameters
        base_image_size: Resize base image to this size
    """

    def __init__(
        self,
        base_image: np.ndarray,
        transform_type: str,
        param_range: np.ndarray,
        base_image_size: int = 128,
    ):
        self.transform_type = transform_type
        self.param_range = param_range

        # Convert to numpy if needed
        if isinstance(base_image, torch.Tensor):
            self.base_image = tensor_to_image(base_image)
        else:
            self.base_image = base_image

        # Resize base image
        import cv2

        self.base_image = cv2.resize(
            base_image,
            (base_image_size, base_image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        self.base_image_size = base_image_size

    def __len__(self) -> int:
        """Number of transformed images."""
        return len(self.param_range)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get transformed image and its parameter.

        Args:
            idx: Index

        Returns:
            image: Transformed image tensor (C, H, W)
            param: Transform parameter value
        """
        from .transforms import ImageTransformer

        param = self.param_range[idx]
        transformer = ImageTransformer(self.base_image)

        if self.transform_type == "rotation":
            img_transformed = transformer.rotate(param)
        elif self.transform_type == "scale":
            img_transformed = transformer.scale(param)
        elif self.transform_type == "translation":
            img_transformed = transformer.translate(param, 0)
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

        # Convert to tensor
        img_tensor = image_to_tensor(img_transformed, normalize=True)

        return img_tensor, param

    def get_all_images(self) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get all transformed images at once.

        Returns:
            images: Tensor of shape (N, C, H, W)
            params: Array of transform parameters
        """
        images = []
        for i in range(len(self)):
            img, param = self[i]
            images.append(img)

        return torch.stack(images), self.param_range


def create_image_grid(
    images: List[np.ndarray], nrow: int = 4
) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Args:
        images: List of images (H, W, C)
        nrow: Number of images per row

    Returns:
        Grid image (H', W', C)
    """
    n = len(images)
    ncol = (n + nrow - 1) // nrow

    h, w = images[0].shape[:2]
    c = images[0].shape[2] if images[0].ndim == 3 else 1

    if c == 1:
        grid = np.zeros((h * ncol, w * nrow), dtype=images[0].dtype)
    else:
        grid = np.zeros((h * ncol, w * nrow, c), dtype=images[0].dtype)

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = img

    return grid


def load_image_pair(
    lr_path: str, hr_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a low-resolution / high-resolution image pair.

    Args:
        lr_path: Path to LR image
        hr_path: Path to HR image

    Returns:
        Tuple of (lr_image, hr_image)
    """
    import cv2

    lr = cv2.imread(lr_path)
    hr = cv2.imread(hr_path)

    if lr is None:
        raise ValueError(f"Failed to load LR image: {lr_path}")
    if hr is None:
        raise ValueError(f"Failed to load HR image: {hr_path}")

    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

    return lr, hr
