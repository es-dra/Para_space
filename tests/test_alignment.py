"""
Synthetic validation of alignment algorithms.

Tests:
1. SIREN: random permutation → align → verify recovery
2. SIREN: check functional invariance after alignment
3. LIIF decoder: random permutation → align → verify recovery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.siren import SIREN
from src.models.liif import LIIFDecoder
from src.alignment import (
    align_siren_parameters,
    align_decoder_parameters,
    AlignmentMethod,
)


def random_permutation(n_neurons):
    """Generate a random permutation of n_neurons."""
    perm = torch.randperm(n_neurons)
    return perm


def apply_permutation_to_siren(siren, layer_idx, perm):
    """Apply neuron permutation to a specific hidden layer of SIREN.

    Permutes the OUTPUT neurons of layer `layer_idx`, which means:
    - W_{layer_idx}: permute rows (outputs)
    - b_{layer_idx}: permute entries
    - W_{layer_idx+1}: permute columns (inputs)
    """
    params = siren.get_params()
    n = len(perm)

    # Permute rows of W_layer and b_layer
    w_key = f"W_{layer_idx}"
    b_key = f"b_{layer_idx}"
    params[w_key] = params[w_key][perm, :]
    params[b_key] = params[b_key][perm]

    # Permute columns of W_{layer_idx+1}
    next_w_key = f"W_{layer_idx + 1}"
    params[next_w_key] = params[next_w_key][:, perm]

    siren.set_params(params)
    return siren


def test_siren_alignment_recovery():
    """Test: create SIREN, permute neurons, align back, verify recovery."""
    torch.manual_seed(42)

    # Create reference SIREN
    siren = SIREN(input_dim=2, hidden_dim=64, num_layers=3, output_dim=3,
                  w0=30.0, w0_initial=30.0, use_siren_init=True)
    theta_ref = siren.get_params()

    # Record reference output
    coords = torch.randn(100, 2)
    with torch.no_grad():
        out_ref = siren(coords).clone()

    # Create a copy and permute layer 0 and layer 1
    siren_perm = SIREN(input_dim=2, hidden_dim=64, num_layers=3, output_dim=3,
                       w0=30.0, w0_initial=30.0, use_siren_init=True)
    siren_perm.set_params({k: v.clone() for k, v in theta_ref.items()})

    perm0 = random_permutation(64)
    perm1 = random_permutation(64)
    apply_permutation_to_siren(siren_perm, 0, perm0)
    apply_permutation_to_siren(siren_perm, 1, perm1)
    theta_perm = siren_perm.get_params()

    # Verify permutation changed parameters
    w0_diff = (theta_ref["W_0"] - theta_perm["W_0"]).abs().max().item()
    assert w0_diff > 1e-6, f"Permutation should change W_0, got max diff {w0_diff}"
    print(f"  [PASS] Permutation changes W_0: max|Δ| = {w0_diff:.6f}")

    # Verify functional output is unchanged (permutation is a symmetry)
    with torch.no_grad():
        out_perm = siren_perm(coords).clone()
    out_diff = (out_ref - out_perm).abs().max().item()
    assert out_diff < 1e-5, f"Permutation should preserve function, got diff {out_diff}"
    print(f"  [PASS] Permutation preserves output: max|Δ| = {out_diff:.2e}")

    # Align permuted back to reference
    theta_aligned, avg_cost = align_siren_parameters(
        theta_ref, theta_perm, method=AlignmentMethod.WEIGHT_MATCHING
    )
    print(f"  Align cost: {avg_cost:.6f}")

    # Verify aligned parameters match reference
    for key in theta_ref:
        diff = (theta_ref[key] - theta_aligned[key]).abs().max().item()
        assert diff < 1e-3, f"Aligned {key} should match reference, got max|Δ| = {diff:.6f}"
    print(f"  [PASS] All aligned params match reference (max|Δ| < 1e-3)")

    # Verify functional output after alignment
    siren_aligned = SIREN(input_dim=2, hidden_dim=64, num_layers=3, output_dim=3,
                          w0=30.0, w0_initial=30.0, use_siren_init=True)
    siren_aligned.set_params(theta_aligned)
    with torch.no_grad():
        out_aligned = siren_aligned(coords).clone()
    mse = torch.nn.functional.mse_loss(out_aligned, out_ref).item()
    psnr = -10 * np.log10(mse + 1e-10)
    assert psnr > 80, f"Aligned SIREN should match reference output (PSNR > 80), got {psnr:.1f}"
    print(f"  [PASS] Aligned output matches reference: PSNR = {psnr:.1f} dB")

    siren_aligned.set_params(theta_aligned)
    print("\n  *** SIREN alignment: ALL CHECKS PASSED ***")
    assert True


def test_liif_decoder_alignment_recovery():
    """Test: create LIIF decoder, permute neurons, align back, verify recovery."""
    torch.manual_seed(123)

    # Create reference decoder: 3 hidden layers (580→256→256→256→3)
    decoder = LIIFDecoder(in_dim=580, hidden_dim=64, num_layers=3, out_dim=3)
    theta_ref = decoder.get_params()

    # Record reference output
    x = torch.randn(50, 580)
    with torch.no_grad():
        out_ref = decoder(x).clone()

    # Manually permute the first hidden layer
    # layers.0.weight: (64, 580), layers.0.bias: (64,)
    # layers.2.weight: (64, 64), layers.2.bias: (64,)
    n_neurons = 64
    perm = random_permutation(n_neurons)

    theta_perm = {k: v.clone() for k, v in theta_ref.items()}
    # Permute layer 0 output neurons
    theta_perm["layers.0.weight"] = theta_perm["layers.0.weight"][perm, :]
    theta_perm["layers.0.bias"] = theta_perm["layers.0.bias"][perm]
    # Permute layer 2 input connections (layer 2 = next Linear after ReLU at index 1)
    theta_perm["layers.2.weight"] = theta_perm["layers.2.weight"][:, perm]

    # Verify permutation changed weights
    w0_diff = (theta_ref["layers.0.weight"] - theta_perm["layers.0.weight"]).abs().max().item()
    assert w0_diff > 1e-6, f"Permutation should change weights, got {w0_diff}"
    print(f"  [PASS] Permutation changes decoder weights: max|Δ| = {w0_diff:.6f}")

    # Verify permuted decoder produces same output (permutation symmetry)
    decoder_perm = LIIFDecoder(in_dim=580, hidden_dim=64, num_layers=3, out_dim=3)
    decoder_perm.set_params(theta_perm)
    with torch.no_grad():
        out_perm = decoder_perm(x).clone()
    out_diff = (out_ref - out_perm).abs().max().item()
    assert out_diff < 1e-5, f"Permutation should preserve function, got {out_diff}"
    print(f"  [PASS] Permutation preserves decoder output: max|Δ| = {out_diff:.2e}")

    # Align permuted back to reference
    theta_aligned, avg_cost = align_decoder_parameters(
        theta_ref, theta_perm, method=AlignmentMethod.WEIGHT_MATCHING
    )
    print(f"  Align cost: {avg_cost:.6f}")

    # Verify aligned parameters match reference
    for key in theta_ref:
        diff = (theta_ref[key] - theta_aligned[key]).abs().max().item()
        assert diff < 1e-3, f"Aligned {key} should match reference, got max|Δ| = {diff:.6f}"
    print(f"  [PASS] All aligned decoder params match reference (max|Δ| < 1e-3)")

    # Verify functional output after alignment
    decoder_aligned = LIIFDecoder(in_dim=580, hidden_dim=64, num_layers=3, out_dim=3)
    decoder_aligned.set_params(theta_aligned)
    with torch.no_grad():
        out_aligned = decoder_aligned(x).clone()
    mse = torch.nn.functional.mse_loss(out_aligned, out_ref).item()
    psnr = -10 * np.log10(mse + 1e-10)
    assert psnr > 80, f"Aligned decoder should match reference (PSNR > 80), got {psnr:.1f}"
    print(f"  [PASS] Aligned decoder output matches reference: PSNR = {psnr:.1f} dB")

    print("\n  *** LIIF decoder alignment: ALL CHECKS PASSED ***")
    assert True


if __name__ == "__main__":
    print("=== SIREN Alignment Validation ===")
    test_siren_alignment_recovery()
    print("\n=== LIIF Decoder Alignment Validation ===")
    test_liif_decoder_alignment_recovery()
    print("\n" + "=" * 50)
    print("ALL ALIGNMENT TESTS PASSED")
    print("=" * 50)
