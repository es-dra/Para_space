"""
Weight permutation alignment for INR parameter spaces.

Handles neuron permutation symmetry for:
1. LIIF/LTE decoder MLPs (layers.{i}.weight/bias format)
2. General state_dict format parameter sets

The decoder MLP in conditional INRs has the same permutation symmetry
as SIREN: hidden neurons can be permuted without changing the function.
This module provides alignment utilities to resolve this ambiguity
before PCA analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn


class AlignmentMethod(Enum):
    NONE = "none"
    WEIGHT_MATCHING = "weight_matching"
    SIGN_FLIP = "sign_flip"


def flatten_params(theta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.flatten().cpu() for p in theta.values()])


def flatten_decoder_params(model: nn.Module) -> torch.Tensor:
    decoder = model.decoder if hasattr(model, 'decoder') else model
    return torch.cat([p.flatten().cpu() for p in decoder.state_dict().values()])


def flatten_encoder_params(model: nn.Module) -> torch.Tensor:
    encoder = model.encoder if hasattr(model, 'encoder') else model
    return torch.cat([p.flatten().cpu() for p in encoder.state_dict().values()])


def _get_decoder_layer_keys(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    pairs = []
    i = 0
    while f"layers.{i}.weight" in state_dict:
        w_key = f"layers.{i}.weight"
        b_key = f"layers.{i}.bias"
        if b_key in state_dict:
            pairs.append((w_key, b_key))
        i += 1
    return pairs


def hungarian_weight_matching(
    W1: torch.Tensor, W2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    h_out, h_in = W1.shape
    W1_expanded = W1.unsqueeze(1)
    W2_expanded = W2.unsqueeze(0)
    C = torch.sum((W1_expanded - W2_expanded) ** 2, dim=2)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C.cpu().numpy())
    perm = torch.tensor(col_ind, dtype=torch.long, device=W1.device)
    W2_permuted = W2[perm, :]
    cost = torch.mean((W1 - W2_permuted) ** 2).item()
    return perm, W2_permuted, cost


def align_sign_flips(
    W1: torch.Tensor, W2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    dots = torch.sum(W1 * W2, dim=1)
    signs = torch.sign(dots)
    signs[signs == 0] = 1
    W2_aligned = W2 * signs.unsqueeze(1)
    cost = torch.mean((W1 - W2_aligned) ** 2).item()
    return W2_aligned, signs, cost


def align_decoder_parameters(
    theta_source: Dict[str, torch.Tensor],
    theta_target: Dict[str, torch.Tensor],
    method: AlignmentMethod = AlignmentMethod.WEIGHT_MATCHING,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Align two decoder parameter sets using permutation symmetry.

    Works with state_dict format: layers.{i}.weight, layers.{i}.bias
    Also handles coord_encoding, scale_encoding, film_layer keys (pass-through).

    Args:
        theta_source: Reference parameter set (state_dict format)
        theta_target: Parameter set to align to source
        method: Alignment method

    Returns:
        theta_target_aligned: Aligned version of theta_target
        total_cost: Average alignment cost across layers
    """
    theta_aligned = {k: v.clone() for k, v in theta_target.items()}

    layer_pairs = _get_decoder_layer_keys(theta_source)
    if not layer_pairs:
        return theta_aligned, 0.0

    num_layers = len(layer_pairs)
    total_cost = 0.0

    permutations = {}
    sign_vectors = {}

    for layer_idx, (w_key, b_key) in enumerate(layer_pairs):
        W_s = theta_source[w_key]
        W_t = theta_target[w_key]

        if W_s.shape != W_t.shape or len(W_s.shape) != 2:
            continue

        is_first = (layer_idx == 0)
        is_last = (layer_idx == num_layers - 1)

        if is_last:
            continue

        if method == AlignmentMethod.WEIGHT_MATCHING:
            perm, W_t_matched, match_cost = hungarian_weight_matching(W_s, W_t)
            W_t_aligned, signs, sign_cost = align_sign_flips(W_s, W_t_matched)
            permutations[layer_idx] = perm
            sign_vectors[layer_idx] = signs
            cost = sign_cost
        elif method == AlignmentMethod.SIGN_FLIP:
            W_t_aligned, signs, cost = align_sign_flips(W_s, W_t)
            sign_vectors[layer_idx] = signs
            permutations[layer_idx] = None
        else:
            W_t_aligned = W_t
            cost = torch.mean((W_s - W_t) ** 2).item()

        theta_aligned[w_key] = W_t_aligned

        if layer_idx in sign_vectors:
            signs = sign_vectors[layer_idx]
            theta_aligned[b_key] = theta_target[b_key] * signs

            if permutations.get(layer_idx) is not None:
                perm = permutations[layer_idx]
                theta_aligned[b_key] = theta_aligned[b_key][perm]

        total_cost += cost

    if num_layers > 1 and method == AlignmentMethod.WEIGHT_MATCHING:
        last_w_key, last_b_key = layer_pairs[-1]
        last_layer_idx = num_layers - 1
        prev_layer_idx = num_layers - 2

        if prev_layer_idx in permutations and permutations[prev_layer_idx] is not None:
            perm = permutations[prev_layer_idx]
            W_t = theta_target[last_w_key]
            theta_aligned[last_w_key] = W_t[:, perm]

    avg_cost = total_cost / max(num_layers - 1, 1)
    return theta_aligned, avg_cost


def align_siren_parameters(
    theta_source: Dict[str, torch.Tensor],
    theta_target: Dict[str, torch.Tensor],
    method: AlignmentMethod = AlignmentMethod.WEIGHT_MATCHING,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Align two SIREN parameter sets using Hungarian matching + sign flip.

    Handles SIREN's sin activation symmetries:
    1. Neuron permutation: hidden neurons can be permuted freely
    2. Sign flip: sin(-x) = -sin(x) — negating a neuron's outgoing weights
       and incoming column is a valid symmetry

    Works with SIREN.get_params() key format: W_0, b_0, ..., W_L, b_L
    where L = num_layers (output layer index).

    Args:
        theta_source: Reference parameter set (from SIREN.get_params())
        theta_target: Parameter set to align to source
        method: AlignmentMethod.WEIGHT_MATCHING or SIGN_FLIP or NONE

    Returns:
        (theta_aligned, average_cost)
    """
    L = 0
    while f"W_{L}" in theta_source:
        L += 1
    n_hidden = L - 1

    theta_aligned = {k: v.clone() for k, v in theta_target.items()}
    permutations = {}
    sign_vectors = {}
    total_cost = 0.0
    valid_layer_count = 0

    for l in range(n_hidden):
        w_key = f"W_{l}"
        b_key = f"b_{l}"
        W_s = theta_source[w_key]
        W_t = theta_target[w_key].clone()

        if W_s.shape[0] != W_s.shape[1]:
            pass

        if l > 0 and (l - 1) in permutations and permutations[l - 1] is not None:
            perm_prev = permutations[l - 1]
            W_t = W_t[:, perm_prev]

        if method == AlignmentMethod.WEIGHT_MATCHING:
            perm, W_t_matched, match_cost = hungarian_weight_matching(W_s, W_t)
            W_t_aligned, signs, sign_cost = align_sign_flips(W_s, W_t_matched)
            permutations[l] = perm
            sign_vectors[l] = signs
            cost = sign_cost
        elif method == AlignmentMethod.SIGN_FLIP:
            W_t_aligned, signs, cost = align_sign_flips(W_s, W_t)
            sign_vectors[l] = signs
            permutations[l] = None
        else:
            W_t_aligned = W_t
            cost = torch.mean((W_s - W_t) ** 2).item()

        theta_aligned[w_key] = W_t_aligned

        b_t = theta_target[b_key].clone()
        if l in permutations and permutations[l] is not None:
            b_t = b_t[permutations[l]]
        if l in sign_vectors:
            b_t = b_t * sign_vectors[l]
        theta_aligned[b_key] = b_t

        next_w_key = f"W_{l + 1}"
        if next_w_key in theta_target and l in permutations:
            W_next = theta_target[next_w_key].clone()
            if l in sign_vectors:
                W_next = W_next * sign_vectors[l].unsqueeze(0)
            if permutations[l] is not None:
                W_next = W_next[:, permutations[l]]
            theta_aligned[next_w_key] = W_next

        total_cost += cost
        valid_layer_count += 1

    avg_cost = total_cost / max(valid_layer_count, 1)
    return theta_aligned, avg_cost


def git_rebasin_align_siren(
    theta_source: Dict[str, torch.Tensor],
    theta_target: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Stub: delegates to align_siren_parameters with weight matching."""
    return align_siren_parameters(
        theta_source, theta_target, method=AlignmentMethod.WEIGHT_MATCHING
    )


def align_decoder_trajectory(
    param_dicts: List[Dict[str, torch.Tensor]],
    method: AlignmentMethod = AlignmentMethod.WEIGHT_MATCHING,
) -> List[Dict[str, torch.Tensor]]:
    if len(param_dicts) == 0:
        return []
    aligned = [param_dicts[0]]
    for i in range(1, len(param_dicts)):
        a, _ = align_decoder_parameters(aligned[-1], param_dicts[i], method=method)
        aligned.append(a)
    return aligned


def get_decoder_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    decoder = model.decoder if hasattr(model, 'decoder') else model
    return {k: v.clone() for k, v in decoder.state_dict().items()}


def get_encoder_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    encoder = model.encoder if hasattr(model, 'encoder') else model
    return {k: v.clone() for k, v in encoder.state_dict().items()}


def set_decoder_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    decoder = model.decoder if hasattr(model, 'decoder') else model
    decoder.load_state_dict(state_dict, strict=True)


def set_encoder_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    encoder = model.encoder if hasattr(model, 'encoder') else model
    encoder.load_state_dict(state_dict, strict=True)


class ParameterSpaceAnalyzer:
    def __init__(self, param_dicts: List[Dict[str, torch.Tensor]]):
        self.param_dicts = param_dicts
        self.flat_params = None
        self.mean = None
        self.components = None
        self.explained_variance = None
        self._compute_flat_params()

    def _compute_flat_params(self):
        self.flat_params = []
        for params in self.param_dicts:
            flat = flatten_params(params)
            self.flat_params.append(flat)
        self.flat_params = torch.stack(self.flat_params)
        self.mean = self.flat_params.mean(dim=0)
        self.flat_params_centered = self.flat_params - self.mean

    def compute_pca(self, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_params = self.flat_params_centered.shape
        q = min(n_components, n_samples - 1)
        from scipy.linalg import svd
        flat_params_cpu = self.flat_params_centered.cpu()
        flat_params_mean_cpu = self.mean.cpu()
        centered = (flat_params_cpu - flat_params_mean_cpu).numpy()
        U, S, Vt = svd(centered, full_matrices=False)
        self.components = Vt.T[:, :n_components]
        self.explained_variance = (S[:n_components] ** 2) / (n_samples - 1)
        self.projections = centered @ Vt.T[:, :n_components]
        return self.components, self.explained_variance

    def project_to_components(self, n: int = 2) -> np.ndarray:
        if self.components is None:
            self.compute_pca(n_components=n)
        return self.projections[:, :n]

    def compute_trajectory_curvature(self) -> float:
        if self.flat_params is None:
            self._compute_flat_params()
        n_samples = self.flat_params.shape[0]
        if n_samples < 3:
            return 0.0
        p0 = self.flat_params[0].numpy()
        p1 = self.flat_params[1].numpy()
        p2 = self.flat_params[2].numpy()
        t1 = p1 - p0
        t2 = p2 - p1
        curvature = np.linalg.norm(t2 - t1) / (np.linalg.norm(t1) + 1e-8)
        return curvature

    def find_trajectory_manifold_dim(self, variance_threshold: float = 0.95) -> int:
        if self.explained_variance is None:
            self.compute_pca(n_components=min(20, len(self.param_dicts) - 1))
        total_var = self.explained_variance.sum()
        cumvar = np.cumsum(self.explained_variance) / total_var
        intrinsic_dim = np.searchsorted(cumvar, variance_threshold) + 1
        return int(intrinsic_dim)
