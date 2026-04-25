"""
Geometric transformations for images.

Provides utilities for creating transformation matrices and applying
geometric transforms to images for the parameter space experiments.
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch


class AffineTransform(Enum):
    """Enumeration of supported affine transform types."""

    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALE = "scale"
    SCALE_ORTHOGONAL = "scale_orthogonal"
    SHEAR = "shear"
    GENERAL = "general"


def create_rotation_matrix(angle_degrees: float) -> np.ndarray:
    """
    Create 2D rotation matrix.

    Args:
        angle_degrees: Rotation angle in degrees

    Returns:
        R: Shape (2, 2) rotation matrix
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)


def create_scale_matrix(
    sx: float, sy: Optional[float] = None
) -> np.ndarray:
    """
    Create 2D scale matrix.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction. If None, use uniform scaling.

    Returns:
        S: Shape (2, 2) scale matrix
    """
    if sy is None:
        sy = sx
    return np.array([[sx, 0], [0, sy]], dtype=np.float32)


def create_translation_matrix(tx: float, ty: float) -> np.ndarray:
    """
    Create 2D translation matrix (as homogeneous affine matrix).

    For homogeneous coordinates:
    T = [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1]]

    Args:
        tx: Translation in x direction
        ty: Translation in y direction

    Returns:
        T: Shape (3, 3) affine transformation matrix in homogeneous coords
    """
    return np.array(
        [[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32
    )


def create_shear_matrix(
    angle: float, direction: str = "horizontal"
) -> np.ndarray:
    """
    Create 2D shear matrix.

    Args:
        angle: Shear angle in degrees
        direction: 'horizontal' or 'vertical'

    Returns:
        S: Shape (2, 2) shear matrix
    """
    angle_rad = np.deg2rad(angle)
    if direction == "horizontal":
        return np.array([[1, np.tan(angle_rad)], [0, 1]], dtype=np.float32)
    else:
        return np.array([[1, 0], [np.tan(angle_rad), 1]], dtype=np.float32)


def decompose_affine_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose affine matrix M = A @ T (linear part + translation).

    For a 3x3 homogeneous matrix:
    M = [[A, b], [0, 1]]

    Args:
        M: Shape (3, 3) affine transformation matrix

    Returns:
        A: Linear part (2, 2)
        b: Translation part (2,)
    """
    A = M[:2, :2]
    b = M[:2, 2]
    return A, b


def apply_geometric_transform(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    interpolation: str = "bilinear",
    output_shape: Optional[Tuple[int, int]] = None,
    border_mode: str = "constant",
    border_value: float = 0.0,
) -> np.ndarray:
    """
    Apply geometric transformation to an image.

    Uses inverse mapping for proper interpolation.

    Args:
        image: Input image (H, W, C) or (H, W)
        transform_matrix: 3x3 affine transformation matrix (for homogeneous coords)
        interpolation: 'bilinear' or 'nearest'
        output_shape: If None, output same shape as input
        border_mode: 'constant', 'reflect', 'replicate', 'wrap'
        border_value: Value for constant border mode

    Returns:
        Transformed image
    """
    import cv2

    h, w = image.shape[:2]
    if output_shape is None:
        output_shape = (h, w)

    # Convert to OpenCV format (if 3x3 homogeneous matrix)
    if transform_matrix.shape == (3, 3):
        M = transform_matrix[:2, :]  # Take first 2 rows
    else:
        M = transform_matrix

    flags = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    border_mode_cv = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    if image.ndim == 3:
        # Color image
        channels = []
        for i in range(image.shape[2]):
            ch = cv2.warpAffine(
                image[:, :, i],
                M,
                (output_shape[1], output_shape[0]),
                flags=flags,
                borderMode=border_mode_cv,
                borderValue=border_value,
            )
            channels.append(ch)
        return np.stack(channels, axis=-1)
    else:
        # Grayscale image
        return cv2.warpAffine(
            image,
            M,
            (output_shape[1], output_shape[0]),
            flags=flags,
            borderMode=border_mode_cv,
            borderValue=border_value,
        )


def apply_inverse_geometric_transform(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear",
    border_mode: str = "constant",
    border_value: float = 0.0,
) -> np.ndarray:
    """
    Apply inverse of geometric transformation.

    For coordinate transform x' = T(x), this computes x = T^(-1)(x').

    Args:
        image: Input image
        transform_matrix: 3x3 transformation matrix
        output_shape: Output shape
        interpolation: 'bilinear' or 'nearest'
        border_mode: Border handling mode
        border_value: Value for constant border

    Returns:
        Transformed image
    """
    import cv2

    h, w = image.shape[:2]
    if output_shape is None:
        output_shape = (h, w)

    # Invert the transformation matrix
    M = transform_matrix[:2, :]
    M_inv = np.linalg.inv(M)

    flags = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    border_mode_cv = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    if image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            ch = cv2.warpAffine(
                image[:, :, i],
                M_inv,
                (output_shape[1], output_shape[0]),
                flags=flags | cv2.WARP_INVERSE_MAP,
                borderMode=border_mode_cv,
                borderValue=border_value,
            )
            channels.append(ch)
        return np.stack(channels, axis=-1)
    else:
        return cv2.warpAffine(
            image,
            M_inv,
            (output_shape[1], output_shape[0]),
            flags=flags | cv2.WARP_INVERSE_MAP,
            borderMode=border_mode_cv,
            borderValue=border_value,
        )


class ImageTransformer:
    """
    Helper class for generating transformed image families.

    Example usage:
        transformer = ImageTransformer(image)
        rotated_family = [transformer.rotate(deg) for deg in range(0, 360, 10)]
        scaled_family = [transformer.scale(factor) for factor in np.arange(0.5, 2.0, 0.1)]

    Args:
        image: Base image (H, W, C) or (H, W)
    """

    def __init__(self, image: np.ndarray):
        self.base_image = image
        self.h, self.w = image.shape[:2]

    def rotate(
        self,
        angle_degrees: float,
        about_center: bool = True,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Rotate image by angle degrees.

        Args:
            angle_degrees: Rotation angle in degrees
            about_center: Whether to rotate about image center
            output_shape: Output shape, if None uses same shape as input

        Returns:
            Rotated image
        """
        import cv2

        if output_shape is None:
            output_shape = (self.w, self.h)

        center = (self.w / 2, self.h / 2) if about_center else (0, 0)
        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        if self.base_image.ndim == 3:
            channels = []
            for i in range(self.base_image.shape[2]):
                ch = cv2.warpAffine(
                    self.base_image[:, :, i],
                    M,
                    output_shape,
                    borderMode=cv2.BORDER_REFLECT,
                )
                channels.append(ch)
            return np.stack(channels, axis=-1)
        else:
            return cv2.warpAffine(
                self.base_image,
                M,
                output_shape,
                borderMode=cv2.BORDER_REFLECT,
            )

    def translate(
        self,
        tx: float,
        ty: float,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Translate image by (tx, ty) pixels.

        Args:
            tx: Translation in x (pixels)
            ty: Translation in y (pixels)
            output_shape: Output shape

        Returns:
            Translated image
        """
        import cv2

        if output_shape is None:
            output_shape = (self.w, self.h)

        M = np.float32([[1, 0, tx], [0, 1, ty]])

        if self.base_image.ndim == 3:
            channels = []
            for i in range(self.base_image.shape[2]):
                ch = cv2.warpAffine(
                    self.base_image[:, :, i], M, output_shape
                )
                channels.append(ch)
            return np.stack(channels, axis=-1)
        else:
            return cv2.warpAffine(self.base_image, M, output_shape)

    def scale(
        self,
        sx: float,
        sy: Optional[float] = None,
        about_center: bool = True,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Scale image by factor(s).

        Args:
            sx: Scale factor in x direction
            sy: Scale factor in y direction. If None, use uniform scaling.
            about_center: Whether to scale about image center
            output_shape: Output shape

        Returns:
            Scaled image
        """
        import cv2

        if sy is None:
            sy = sx
        if output_shape is None:
            output_shape = (int(self.w * sx), int(self.h * sy))

        if about_center:
            center = (self.w / 2, self.h / 2)
            M = cv2.getRotationMatrix2D(center, 0, sx)
            # Adjust translation for center-based scaling
            M[0, 2] += output_shape[0] / 2 - center[0] * sx
            M[1, 2] += output_shape[1] / 2 - center[1] * sy
        else:
            M = np.array(
                [[sx, 0, 0], [0, sy, 0]], dtype=np.float32
            )

        if self.base_image.ndim == 3:
            channels = []
            for i in range(self.base_image.shape[2]):
                ch = cv2.warpAffine(
                    self.base_image[:, :, i],
                    M,
                    output_shape,
                    borderMode=cv2.BORDER_REFLECT,
                )
                channels.append(ch)
            return np.stack(channels, axis=-1)
        else:
            return cv2.warpAffine(
                self.base_image,
                M,
                output_shape,
                borderMode=cv2.BORDER_REFLECT,
            )

    def affine(self, matrix: np.ndarray, output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply arbitrary affine transformation.

        Args:
            matrix: 2x3 or 3x3 affine transformation matrix
            output_shape: Output shape

        Returns:
            Transformed image
        """
        if matrix.shape == (3, 3):
            M = matrix[:2, :]
        else:
            M = matrix

        if output_shape is None:
            output_shape = (self.w, self.h)

        return apply_geometric_transform(self.base_image, M, output_shape=output_shape)

    def get_transform_matrices(
        self,
        transform_type: str,
        param_range: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get (A, b) pairs for a range of transform parameters.

        Args:
            transform_type: 'rotation', 'scale', or 'translation'
            param_range: Array of transform parameters

        Returns:
            List of (linear_part, translation_part) tuples
        """
        result = []
        for param in param_range:
            if transform_type == "rotation":
                R = create_rotation_matrix(param)
                result.append((R, np.zeros(2, dtype=np.float32)))
            elif transform_type == "scale":
                S = create_scale_matrix(param)
                result.append((S, np.zeros(2, dtype=np.float32)))
            elif transform_type == "translation":
                T = np.eye(2, dtype=np.float32)
                b = np.array([param, 0], dtype=np.float32)
                result.append((T, b))
        return result


def create_rotation_transform_matrices(
    angles: List[float],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create rotation transformation matrices for a list of angles.

    Args:
        angles: List of rotation angles in degrees

    Returns:
        List of (R, b) tuples where R is rotation matrix and b is zero translation
    """
    return [(create_rotation_matrix(a), np.zeros(2, dtype=np.float32)) for a in angles]


def create_scale_transform_matrices(
    scales: List[float],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create scale transformation matrices for a list of scale factors.

    Args:
        scales: List of scale factors

    Returns:
        List of (S, b) tuples where S is scale matrix and b is zero translation
    """
    return [(create_scale_matrix(s), np.zeros(2, dtype=np.float32)) for s in scales]
