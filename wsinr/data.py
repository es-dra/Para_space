"""MNIST from the raw idx files, and simple synthetic shapes."""
from __future__ import annotations

import gzip
import os
import struct

import numpy as np

MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist(root: str, split: str = "train"):
    """Returns images float32 in [0, 1] of shape (N, 28, 28) and int labels (N,)."""
    xi, yi = MNIST_FILES[split]
    x = _read_idx(os.path.join(root, xi)).astype(np.float32) / 255.0
    y = _read_idx(os.path.join(root, yi)).astype(np.int64)
    return x, y


def shapes_dataset(n: int, side: int = 28, seed: int = 0):
    """Filled circles / squares / triangles at random size and position (labels 0/1/2)."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, side, side), np.float32)
    labels = rng.integers(0, 3, size=n)
    yy, xx = np.mgrid[0:side, 0:side]
    for i, lab in enumerate(labels):
        r = rng.integers(side // 6, side // 3)
        cy, cx = rng.integers(r + 1, side - r - 1, size=2)
        if lab == 0:
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        elif lab == 1:
            m = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
        else:
            m = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= (yy - (cy - r)) / 2.0)
        imgs[i][m] = 1.0
    return imgs, labels
