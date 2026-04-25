"""
SIREN (Sinusoidal Representation Network) implementation.

SIRENs use sin activation functions instead of ReLU/tanh, enabling
exact representation of coordinate-based functions with expressive gradients.

Key symmetries:
- Sign flip: sin(-z) = -sin(z) -> flip input/output weights of a neuron
- Periodic shift: sin(z + 2πk) = sin(z) -> bias shift invariance
- Half-period: sin(z + π) = -sin(z) -> bias shift equivalence to sign flip
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SIREN(nn.Module):
    """
    SIREN: Implicit Neural Representation with sin activation.

    Architecture: f(x) = W_L σ(W_{L-1} ... σ(W_1 x + b_1) ... + b_{L-1}) + b_L
    where σ(z) = sin(z)

    Args:
        input_dim: Coordinate dimension (2 for 2D images)
        hidden_dim: Hidden layer width
        num_layers: Number of hidden layers
        output_dim: Output dimension (3 for RGB)
        w0: Frequency parameter for sin activation
        w0_initial: Frequency for first layer
        use_siren_init: Use SIREN's sinusoidal initialization
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 3,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        use_siren_init: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.w0 = w0
        self.w0_initial = w0_initial

        # Build layers
        layers = []
        for i in range(num_layers + 1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
                if use_siren_init:
                    self._siren_init_layer(layer, w0_initial, is_first=True)
            elif i == num_layers:
                layer = nn.Linear(hidden_dim, output_dim)
                if use_siren_init:
                    self._siren_init_layer(layer, w0_initial, is_first=False)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
                if use_siren_init:
                    self._siren_init_layer(layer, w0, is_first=False)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def _siren_init_layer(self, layer: nn.Linear, w0: float, is_first: bool = False):
        """
        Initialize layer weights using SIREN's sinusoidal initialization.

        From Sitzmann et al. (2020):
        - First layer: b = 0, W ~ N(0, 1/w0²)
        - Hidden layers: b ~ N(0, 1), W ~ N(0, 1/w0²)
        - Output layer: b = 0, W ~ N(0, 1)

        Args:
            layer: Linear layer to initialize
            w0: Frequency parameter
            is_first: Whether this is the first layer
        """
        with torch.no_grad():
            if is_first:
                # First layer: W ~ N(0, (1/w0)²), b = 0
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / w0)
                nn.init.zeros_(layer.bias)
            elif layer.out_features == self.output_dim:
                # Output layer: W ~ N(0, 1), b = 0
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.zeros_(layer.bias)
            else:
                # Hidden layers: W ~ N(0, (1/w0)²), b ~ N(0, 1)
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / w0)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN.

        Args:
            coords: Shape (N, input_dim) or (B, N, input_dim)

        Returns:
            output: Shape (N, output_dim) or (B, N, output_dim)
        """
        x = coords
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.sin(x)
        # Final layer without activation
        x = self.layers[-1](x)
        return x

    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Return all parameters as a dictionary.

        Returns:
            Dict with keys 'W_0', 'b_0', 'W_1', 'b_1', ..., 'W_L', 'b_L'
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"W_{i}"] = layer.weight.data.clone()
            params[f"b_{i}"] = layer.bias.data.clone()
        return params

    def set_params(self, params: Dict[str, torch.Tensor]):
        """
        Set parameters from a dictionary.

        Args:
            params: Dict with keys 'W_0', 'b_0', 'W_1', 'b_1', ...
        """
        for i, layer in enumerate(self.layers):
            layer.weight.data = params[f"W_{i}"].clone()
            layer.bias.data = params[f"b_{i}"].clone()

    def first_layer_transform(self, A: torch.Tensor, b: torch.Tensor) -> "SIREN":
        """
        Apply first-layer transformation theorem.

        Given input transform x' = A @ x + b, the equivalent SIREN parameter
        transformation is:
        - W_0' = W_0 @ A (assuming W_0 is the first layer weight)
        - b_0' = b_0 + W_0 @ b
        - W_l' = W_l, b_l' = b_l for l >= 1

        This is an EXACT algebraic identity for any activation function σ,
        not just sin. It works because:
        σ(W_0 x + b_0) = σ(W_0 (A x + b) + (b_0 - W_0 @ b))
                      = σ(W_0' x + b_0')

        Args:
            A: Shape (input_dim, input_dim) - linear part of transform
            b: Shape (input_dim,) - translation part of transform

        Returns:
            New SIREN with transformed first layer
        """
        # Create a new SIREN with copied parameters
        new_siren = SIREN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            w0=self.w0,
            w0_initial=self.w0_initial,
            use_siren_init=False,  # We'll set params manually
        )

        # Copy all parameters
        original_params = self.get_params()
        new_params = {}
        for key, value in original_params.items():
            new_params[key] = value.clone()

        # Apply first-layer transformation
        # W_0' = W_0 @ A
        W_0 = original_params["W_0"]  # (hidden_dim, input_dim)
        A_tensor = A.to(W_0.device, W_0.dtype)  # (input_dim, input_dim)
        new_params["W_0"] = W_0 @ A_tensor

        # b_0' = b_0 + W_0 @ b
        b_tensor = b.to(W_0.device, W_0.dtype)  # (input_dim,)
        new_params["b_0"] = original_params["b_0"] + W_0 @ b_tensor

        new_siren.set_params(new_params)
        return new_siren

    def __repr__(self):
        return (
            f"SIREN(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, output_dim={self.output_dim}, "
            f"w0={self.w0})"
        )


def compute_first_layer_analytic(
    theta_original: Dict[str, torch.Tensor],
    A: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute analytically transformed parameters using first-layer theorem.

    Given original parameters θ and transform (A, b), compute:
    - W_0' = W_0 @ A
    - b_0' = b_0 + W_0 @ b
    - W_l' = W_l, b_l' = b_l for l >= 1

    This is an EXACT algebraic identity.

    Args:
        theta_original: Dict with keys 'W_0', 'b_0', 'W_1', 'b_1', etc.
        A: Shape (input_dim, input_dim) - linear part of transform
        b: Shape (input_dim,) - translation part of transform

    Returns:
        theta_analytic: Dict with transformed parameters
    """
    theta_new = {k: v.clone() for k, v in theta_original.items()}

    W_0 = theta_original["W_0"]  # (hidden_dim, input_dim)
    b_0 = theta_original["b_0"]  # (hidden_dim,)

    # W_0' = W_0 @ A
    # A is (input_dim, input_dim)
    # W_0 @ A = (hidden_dim, input_dim) @ (input_dim, input_dim) = (hidden_dim, input_dim)
    theta_new["W_0"] = W_0 @ A

    # b_0' = b_0 + W_0 @ b
    # b is (input_dim,), W_0 @ b = (hidden_dim, input_dim) @ (input_dim,) = (hidden_dim,)
    theta_new["b_0"] = b_0 + W_0 @ b

    return theta_new


def get_siren_param_count(
    num_layers: int,
    hidden_dim: int,
    input_dim: int = 2,
    output_dim: int = 3,
) -> int:
    """
    Calculate total number of trainable parameters in SIREN.

    For L hidden layers with width h, input dim d_in, output dim d_out:
    Total = d_in*h + h + (L-1)*h*h + (L-1)*h + h*d_out + d_out
         = d_in*h + L*h + (L-1)*h^2 + h*d_out + d_out

    Args:
        num_layers: Number of hidden layers
        hidden_dim: Width of hidden layers
        input_dim: Input dimension
        output_dim: Output dimension

    Returns:
        Total parameter count
    """
    # First layer: input_dim * hidden_dim + hidden_dim
    total = input_dim * hidden_dim + hidden_dim
    # Hidden layers (num_layers - 1): hidden_dim * hidden_dim + hidden_dim each
    total += (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    # Output layer: hidden_dim * output_dim + output_dim
    total += hidden_dim * output_dim + output_dim
    return total


def create_siren_layer(
    input_dim: int,
    output_dim: int,
    w0: float = 30.0,
    is_first_layer: bool = False,
    use_siren_init: bool = True,
) -> Tuple[nn.Linear, nn.Parameter]:
    """
    Create a single SIREN layer with proper initialization.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        w0: Frequency parameter
        is_first_layer: Whether this is the first layer
        use_siren_init: Use SIREN initialization

    Returns:
        Tuple of (linear_module, bias_parameter)
    """
    layer = nn.Linear(input_dim, output_dim)

    if use_siren_init:
        if is_first_layer:
            nn.init.uniform_(layer.weight, -w0 / input_dim, w0 / input_dim)
            nn.init.zeros_(layer.bias)
        else:
            nn.init.uniform_(layer.weight, -w0 / output_dim, w0 / output_dim)
            nn.init.normal_(layer.bias, 0, 1)

    return layer, layer.bias
