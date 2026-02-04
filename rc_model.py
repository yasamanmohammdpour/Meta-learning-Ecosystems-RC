# rc_model.py

import torch
import torch.nn as nn


class Reservoir(nn.Module):
    """
    Exact PyTorch replica of func_rc_train_val reservoir.

    - Fixed W_in and A
    - Sparse symmetric reservoir
    - Spectral radius scaling
    - Leaky integrator
    - Trainable W_out only
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

        # --------------------------------------------------
        # W_in ~ Uniform[-W_in_a, W_in_a]
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
        # Trainable readout (Wout)
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

    # ======================================================
    # State handling
    # ======================================================
    def reset_state(self):
        self.r.zero_()

    # ======================================================
    # One-step reservoir update
    # EXACT match to:
    # r_new = (1-α) r + α tanh(A r + W_in x)
    # ======================================================
    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)

        self.r.mul_(1.0 - self.alpha).add_(
            self.alpha * torch.tanh(self.A @ self.r + self.W_in @ x)
        )

        return self.r.clone()

    # ======================================================
    # Readout
    # ======================================================
    def readout(self, r: torch.Tensor) -> torch.Tensor:
        return self.W_out(r)
