"""
Utility functions for INR experiments.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get torch device.

    Args:
        gpu_id: GPU device ID. If -1 or negative, use CPU.

    Returns:
        torch.device: CUDA device if available and gpu_id >= 0, else CPU
    """
    if gpu_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """
    Compute and store running average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LearningRateScheduler:
    """
    Custom learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        min_lr: Minimum learning rate after decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = max(self.min_lr, self.base_lrs[i] * factor)

        self.current_epoch += 1

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def lbfgs_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.LBFGS,
    coords: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 20,
) -> float:
    """
    Single L-BFGS optimization step for INR training.

    L-BFGS is preferred for INR training as it converges faster than Adam.

    Args:
        model: SIREN model
        optimizer: L-BFGS optimizer
        coords: Coordinate grid (N, input_dim)
        targets: Target values (N, output_dim)
        max_iter: Maximum L-BFGS iterations

    Returns:
        Loss value
    """
    closure_count = [0]

    def closure():
        optimizer.zero_grad()
        output = model(coords)
        loss = torch.mean((output - targets) ** 2)
        loss.backward()
        closure_count[0] += 1
        return loss

    optimizer.step(closure)
    return closure_count[0]


def adam_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    coords: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Single Adam optimization step for INR training.

    Args:
        model: SIREN model
        optimizer: Adam optimizer
        coords: Coordinate grid (N, input_dim)
        targets: Target values (N, output_dim)

    Returns:
        Loss value
    """
    optimizer.zero_grad()
    output = model(coords)
    loss = torch.mean((output - targets) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss value
        path: Save path
        **kwargs: Additional items to save (e.g., scheduler)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path

    Returns:
        Checkpoint dictionary
    """
    return torch.load(path, map_location="cpu")
