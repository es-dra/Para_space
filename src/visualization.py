"""
Visualization utilities for INR parameter space experiments.

Provides plotting functions for:
- Parameter trajectory PCA visualization
- Perturbation effect comparison
- Layer-wise parameter differences
- Semantic clustering
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_parameter_trajectory_2d(
    projections: np.ndarray,
    transform_params: List[float],
    transform_type: str,
    title: str,
    save_path: Optional[str] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot 2D PCA projection of parameter trajectory.

    Args:
        projections: Shape (n_samples, 2)
        transform_params: List of transform parameters (e.g., rotation angles)
        transform_type: 'rotation', 'scale', 'translation'
        title: Plot title
        save_path: If provided, save figure to this path
        labels: Optional labels for each point
        colors: Optional colors for each point
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap based on transform parameters
    param_array = np.array(transform_params)
    norm = plt.Normalize(vmin=param_array.min(), vmax=param_array.max())

    if colors is None:
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=param_array,
            cmap="viridis",
            norm=norm,
            s=100,
            alpha=0.8,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        if transform_type == "rotation":
            cbar.set_label("Rotation (degrees)")
        elif transform_type == "scale":
            cbar.set_label("Scale factor")
        elif transform_type == "translation":
            cbar.set_label("Translation (pixels)")
    else:
        ax.scatter(projections[:, 0], projections[:, 1], c=colors, s=100, alpha=0.8)

    # Draw trajectory lines
    for i in range(len(projections) - 1):
        ax.plot(
            [projections[i, 0], projections[i + 1, 0]],
            [projections[i, 1], projections[i + 1, 1]],
            "k-",
            alpha=0.3,
            linewidth=1,
        )
        # Add arrows
        dx = projections[i + 1, 0] - projections[i, 0]
        dy = projections[i + 1, 1] - projections[i, 1]
        ax.annotate(
            "",
            xy=(projections[i + 1, 0], projections[i + 1, 1]),
            xytext=(projections[i, 0], projections[i, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    # Mark start and end
    ax.scatter(
        [projections[0, 0]], [projections[0, 1]], c="green", s=200, marker="*", zorder=5, label="Start"
    )
    ax.scatter(
        [projections[-1, 0]], [projections[-1, 1]], c="red", s=200, marker="*", zorder=5, label="End"
    )

    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (projections[i, 0], projections[i, 1]), fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_parameter_trajectory_3d(
    projections: np.ndarray,
    transform_params: np.ndarray,
    transform_type: str,
    title: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    3D version of trajectory plotting.

    Args:
        projections: Shape (n_samples, 3)
        transform_params: Array of transform parameters
        transform_type: Type of transformation
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create colormap
    norm = plt.Normalize(vmin=transform_params.min(), vmax=transform_params.max())
    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        projections[:, 2],
        c=transform_params,
        cmap="viridis",
        norm=norm,
        s=100,
        alpha=0.8,
    )

    # Draw trajectory lines
    for i in range(len(projections) - 1):
        ax.plot(
            [projections[i, 0], projections[i + 1, 0]],
            [projections[i, 1], projections[i + 1, 1]],
            [projections[i, 2], projections[i + 1, 2]],
            "k-",
            alpha=0.3,
            linewidth=1,
        )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    if transform_type == "rotation":
        cbar.set_label("Rotation (degrees)")
    elif transform_type == "scale":
        cbar.set_label("Scale factor")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_perturbation_effects(
    original_image: np.ndarray,
    perturbed_params_list: List[Tuple[Dict[str, torch.Tensor], np.ndarray]],
    perturbation_descriptions: List[str],
    save_path: Optional[str] = None,
    max_display: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Visualize effects of different parameter perturbations on output.

    Creates a grid showing original image and each perturbation result.

    Args:
        original_image: Original image (H, W, C)
        perturbed_params_list: List of (perturbed_params, perturbed_image) tuples
        perturbation_descriptions: Descriptions for each perturbation
        save_path: If provided, save figure
        max_display: Maximum number of perturbations to display
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_display = min(len(perturbed_params_list), max_display)
    if figsize is None:
        figsize = (3 * (n_display + 1), 3)

    fig, axes = plt.subplots(1, n_display + 1, figsize=figsize)

    # Plot original
    axes[0].imshow(np.clip(original_image, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot perturbations
    for i in range(n_display):
        params, img = perturbed_params_list[i]
        axes[i + 1].imshow(np.clip(img, 0, 1))
        axes[i + 1].set_title(perturbation_descriptions[i][:20])
        axes[i + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layerwise_param_diff(
    theta1: Dict[str, torch.Tensor],
    theta2: Dict[str, torch.Tensor],
    title: str = "Layer-wise Parameter Difference",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot per-layer parameter difference between two SIREN models.

    Useful for P1 experiment to verify first-layer transformation theorem.

    Args:
        theta1: First parameter set
        theta2: Second parameter set
        title: Plot title
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    layers = []
    weight_diffs = []
    bias_diffs = []

    i = 0
    while f"W_{i}" in theta1 and f"W_{i}" in theta2:
        layers.append(f"Layer {i}")

        W_diff = torch.mean((theta1[f"W_{i}"] - theta2[f"W_{i}"]) ** 2).item()
        weight_diffs.append(W_diff)

        b_diff = torch.mean((theta1[f"b_{i}"] - theta2[f"b_{i}"]) ** 2).item()
        bias_diffs.append(b_diff)

        i += 1

    x = np.arange(len(layers))
    width = 0.35

    ax1.bar(x - width / 2, weight_diffs, width, label="Weight MSE")
    ax1.bar(x + width / 2, bias_diffs, width, label="Bias MSE")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("MSE")
    ax1.set_title("Per-Layer MSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Summary statistics
    total_diff = np.sum(weight_diffs) + np.sum(bias_diffs)
    ax2.text(
        0.5,
        0.5,
        f"Total MSE: {total_diff:.6f}\n"
        + f"Max Weight MSE: {max(weight_diffs):.6f}\n"
        + f"Max Bias MSE: {max(bias_diffs):.6f}",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax2.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_equivariance_error(
    layer_equiv_errors: Dict[int, float],
    title: str = "Equivariance Error by Layer",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot equivariance error as function of layer depth (for P4 experiment).

    Args:
        layer_equiv_errors: Dict mapping layer index to equivariance error
        title: Plot title
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = sorted(layer_equiv_errors.keys())
    errors = [layer_equiv_errors[l] for l in layers]

    ax.plot(layers, errors, "bo-", linewidth=2, markersize=8)
    ax.fill_between(layers, errors, alpha=0.3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Equivariance Error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_semantic_clustering(
    param_projections: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    label_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot parameter PCA projections with semantic labels (for P3b).

    Different digit classes should form separate clusters if
    parameter space has semantic structure.

    Args:
        param_projections: Shape (n_samples, 2)
        labels: Semantic labels for each point (e.g., "digit_3", "digit_7")
        title: Plot title
        save_path: If provided, save figure
        label_colors: Optional dict mapping labels to colors
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels
    unique_labels = sorted(set(labels))

    if label_colors is None:
        # Generate colors
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = [l == label for l in labels]
        ax.scatter(
            param_projections[mask, 0],
            param_projections[mask, 1],
            c=[label_colors[label]],
            label=label,
            s=100,
            alpha=0.7,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_explained_variance(
    explained_variance: np.ndarray,
    title: str = "Explained Variance by Principal Component",
    variance_threshold: float = 0.95,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot explained variance for PCA components.

    Args:
        explained_variance: Array of explained variance per component
        title: Plot title
        variance_threshold: Draw horizontal line at this cumulative threshold
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_components = len(explained_variance)
    x = np.arange(1, n_components + 1)

    # Individual variance
    ax1.bar(x, explained_variance, alpha=0.7)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance")
    ax1.set_title("Variance per Component")
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumvar = np.cumsum(explained_variance) / explained_variance.sum()
    ax2.plot(x, cumvar, "bo-", linewidth=2, markersize=8)
    ax2.axhline(y=variance_threshold, color="r", linestyle="--", label=f"{variance_threshold:.0%}")
    ax2.fill_between(x, cumvar, alpha=0.3)

    # Mark intrinsic dimension
    intrinsic_dim = np.searchsorted(cumvar, variance_threshold) + 1
    ax2.axvline(x=intrinsic_dim, color="g", linestyle="--", label=f"Dim = {intrinsic_dim}")

    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot training loss curve.

    Args:
        losses: List of loss values per iteration
        title: Plot title
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(losses, linewidth=1, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_trajectory_animation(
    projections: np.ndarray,
    transform_params: np.ndarray,
    transform_type: str,
    output_path: str,
    fps: int = 10,
):
    """
    Create animated visualization of parameter trajectory.

    Requires imageio to be installed.

    Args:
        projections: Shape (n_samples, 2)
        transform_params: Array of transform parameters
        transform_type: Type of transformation
        output_path: Output path for GIF
        fps: Frames per second
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed, skipping animation")
        return

    frames = []
    n_samples = len(projections)

    for i in range(0, n_samples, max(1, n_samples // 50)):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all points up to current index
        scatter = ax.scatter(
            projections[: i + 1, 0],
            projections[: i + 1, 1],
            c=transform_params[: i + 1],
            cmap="viridis",
            s=100,
            alpha=0.8,
        )

        # Draw trajectory
        if i > 0:
            ax.plot(
                projections[: i + 1, 0],
                projections[: i + 1, 1],
                "k-",
                alpha=0.3,
                linewidth=1,
            )

        # Mark current point
        ax.scatter(
            [projections[i, 0]],
            [projections[i, 1]],
            c="red",
            s=200,
            marker="*",
            zorder=5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        if transform_type == "rotation":
            cbar.set_label("Rotation (degrees)")
        elif transform_type == "scale":
            cbar.set_label("Scale factor")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Parameter Trajectory (Frame {i + 1}/{n_samples})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)


def plot_psnr_vs_epsilon(
    perturbations: Dict[str, Dict],
    title: str = "PSNR vs Epsilon for Different Perturbation Types",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot PSNR vs epsilon for different perturbation strategies.

    Used in P2 experiment to analyze parameter space geometry.

    Args:
        perturbations: Dict mapping perturbation type to dict with 'epsilons' and 'psnrs'
        title: Plot title
        save_path: If provided, save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    perturbation_colors = {
        "random": "gray",
        "first_layer_rotation": "blue",
        "first_layer_scale": "green",
        "bias": "orange",
        "deep_layer": "red",
    }

    perturbation_labels = {
        "random": "Random perturbation",
        "first_layer_rotation": "First-layer rotation direction",
        "first_layer_scale": "First-layer scale direction",
        "bias": "Bias perturbation (b_0)",
        "deep_layer": "Deep layer perturbation (W_2, b_2)",
    }

    for pert_name, pert_data in perturbations.items():
        epsilons = pert_data["epsilons"]
        psnrs = pert_data["psnrs"]

        color = perturbation_colors.get(pert_name, None)
        label = perturbation_labels.get(pert_name, pert_name)

        if color:
            ax.plot(epsilons, psnrs, "o-", color=color, label=label, linewidth=2, markersize=6)
        else:
            ax.plot(epsilons, psnrs, "o-", label=pert_name, linewidth=2, markersize=6)

    ax.set_xlabel("Epsilon (perturbation magnitude)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
