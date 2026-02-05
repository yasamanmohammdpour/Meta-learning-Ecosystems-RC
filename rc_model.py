# rc_model.py
import torch
import torch.nn as nn


class Reservoir(nn.Module):
    """
    OPTIMIZED PyTorch Reservoir Computing model.
    
    Key features:
    - Fixed W_in and A (not trained)
    - Sparse symmetric reservoir
    - Spectral radius scaling
    - Leaky integrator dynamics
    - Trainable W_out only (via ridge regression)
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        output_dim: int,
        alpha: float,
        sparsity: float,
        spectral_radius: float,
        w_in_scale: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype

        self.n = reservoir_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        # Pre-compute constants for speed
        self.register_buffer("alpha_tensor", torch.tensor(alpha, device=device, dtype=dtype))
        self.register_buffer("one_minus_alpha", torch.tensor(1.0 - alpha, device=device, dtype=dtype))

        # --------------------------------------------------
        # W_in ~ Uniform[-w_in_scale, w_in_scale]
        # --------------------------------------------------
        W_in = w_in_scale * (
            2.0 * torch.rand(self.n, input_dim, device=device, dtype=dtype) - 1.0
        )
        self.register_buffer("W_in", W_in)

        # --------------------------------------------------
        # Sparse symmetric reservoir A
        # --------------------------------------------------
        A = torch.zeros(self.n, self.n, device=device, dtype=dtype)

        mask = torch.rand(self.n, self.n, device=device) < sparsity
        A[mask] = torch.randn(mask.sum(), device=device, dtype=dtype)

        # Enforce symmetry: A = (S + Sᵀ)/2
        A = 0.5 * (A + A.T)

        # Spectral radius scaling
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # symmetric → real eigenvalues
            sr = torch.max(torch.abs(eigvals))
            if sr > 0:
                A.mul_(spectral_radius / sr)

        self.register_buffer("A", A)

        # --------------------------------------------------
        # Trainable readout (W_out)
        # --------------------------------------------------
        self.W_out = nn.Linear(
            self.n, output_dim, bias=False, device=device, dtype=dtype
        )

        # --------------------------------------------------
        # Reservoir state r
        # --------------------------------------------------
        self.register_buffer(
            "r", torch.zeros(self.n, device=device, dtype=dtype)
        )

    def reset_state(self):
        """Reset reservoir state to zero."""
        self.r.zero_()

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-step reservoir update: r = (1-α)r + α*tanh(A*r + W_in*x)
        OPTIMIZED: No clone, pre-computed constants
        """
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype)
        
        # Compute: A @ r + W_in @ x
        Ar = self.A @ self.r
        Win_x = self.W_in @ x
        
        # Update in-place
        self.r.mul_(self.one_minus_alpha).add_(
            self.alpha_tensor * torch.tanh(Ar + Win_x)
        )
        
        return self.r

    def readout(self, r: torch.Tensor) -> torch.Tensor:
        """Apply readout layer to reservoir state(s)."""
        return self.W_out(r)