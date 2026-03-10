"""BabyBrain — The randomly initialized neural network at EVA's core.

Ron Protocol: NO pretrained weights. Every EVA starts from scratch.
Attempts to use Mamba SSM if available, falls back to Transformer.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def detect_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Try to import Mamba SSM
_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba

    _MAMBA_AVAILABLE = True
    logger.info("Mamba SSM available — using Mamba architecture.")
except ImportError:
    logger.info("Mamba SSM not available — using Transformer architecture.")


class MambaBlock(nn.Module):
    """Wrapper around Mamba SSM block with layer norm."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class BabyBrain(nn.Module):
    """The core neural network of an EVA.

    RANDOMLY INITIALIZED. No pretrained weights. This is non-negotiable.
    Uses Mamba blocks if mamba_ssm is installed, otherwise falls back
    to standard Transformer encoder blocks.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Hidden dimension size.
        n_layers: Number of encoder layers.
        n_heads: Number of attention heads (transformer only).
        dtype_str: Parameter dtype as string ("float16" or "float32").
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dtype_str: str = "float16",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dtype = torch.float16 if dtype_str == "float16" else torch.float32
        self.device = device if device is not None else detect_device()
        self.architecture = "mamba" if _MAMBA_AVAILABLE else "transformer"

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 2048, d_model) * 0.02
        )

        # Encoder layers
        if _MAMBA_AVAILABLE:
            self.layers = nn.ModuleList(
                [MambaBlock(d_model) for _ in range(n_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
            )

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Store last hidden state
        self._last_hidden: Optional[torch.Tensor] = None

        # Convert to target dtype and move to device
        self.to(dtype=self.dtype, device=self.device)

        # Log architecture info
        logger.info(
            "BabyBrain initialized: arch=%s, params=%d, "
            "est_memory=%.3f GB, d_model=%d, n_layers=%d, device=%s",
            self.architecture,
            self.parameter_count,
            self._estimate_memory_gb(),
            d_model,
            n_layers,
            self.device,
        )

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _estimate_memory_gb(self) -> float:
        """Estimate memory usage in GB."""
        bytes_per_param = 2 if self.dtype == torch.float16 else 4
        return (self.parameter_count * bytes_per_param) / (1024 ** 3)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Tuple of (logits, hidden_state):
                logits: Shape (batch, seq_len, vocab_size)
                hidden_state: Shape (batch, seq_len, d_model)
        """
        seq_len = input_ids.shape[-1]

        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Encoder layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        hidden = self.norm(x)
        self._last_hidden = hidden.detach()

        # Output projection
        logits = self.output_proj(hidden)

        return logits, hidden

    def predict_next(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict probability distribution over next token.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Probability distribution of shape (batch, vocab_size).
        """
        logits, _ = self.forward(input_ids)
        # Take logits for the last position
        last_logits = logits[:, -1, :]
        return F.softmax(last_logits.float(), dim=-1)

    def get_hidden_state(self) -> torch.Tensor:
        """Return the last hidden state from the most recent forward pass.

        Returns:
            Hidden state tensor, or zeros if no forward pass yet.
        """
        if self._last_hidden is not None:
            return self._last_hidden
        return torch.zeros(1, 1, self.d_model)

    def get_parameter_snapshot(
        self, sample_ratio: float = 1.0
    ) -> dict[str, dict[str, float]]:
        """Lightweight parameter snapshot — mean and std per layer.

        Used for information gain computation. NOT a full copy.

        Args:
            sample_ratio: Fraction of parameters to sample (0.0-1.0).
                Use < 1.0 for faster snapshots at the cost of precision.

        Returns:
            Dict mapping layer name to {"mean": float, "std": float}.
        """
        snapshot: dict[str, dict[str, float]] = {}
        params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]

        if sample_ratio < 1.0:
            import random
            k = max(1, int(len(params) * sample_ratio))
            params = random.sample(params, k)

        for name, param in params:
            with torch.no_grad():
                p = param.float()
                snapshot[name] = {
                    "mean": p.mean().item(),
                    "std": p.std().item(),
                }
        return snapshot
