# 1D Convolutional Neural Network for Exoplanet Detection
# Architecture inspired by Shallue & Vanderburg (2018) "Identifying Exoplanets
# with Deep Learning" — the Google Brain paper that detected 2 new exoplanets
# using this exact approach on Kepler data.
# Input  : phase-folded, binned light curve  → shape (batch, 1, 201)
# Output : class logits                       → shape (batch, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ConvBlock(nn.Module):
    """
    Reusable building block: Conv1d → BatchNorm → ReLU → MaxPool

    Why each layer?

    ▸ Conv1d(in_ch, out_ch, kernel_size)
      Slides a learnable filter of width `kernel_size` across the 201-point
      curve. Each filter detects a local pattern (e.g., a smooth U-shaped dip).
      We stack multiple filters (out_ch) so the layer learns diverse features.

    ▸ BatchNorm1d
      Normalises activations within each mini-batch so:
        - Training is more stable (less sensitive to weight initialisation)
        - We can use higher learning rates
        - Acts as a mild regulariser

    ▸ ReLU
      Non-linear activation. Without it, stacking conv layers would collapse
      into a single linear transformation — the network couldn't learn
      complex transit shapes.

    ▸ MaxPool1d(kernel_size=2, stride=2)
      Halves the sequence length at each block, creating a hierarchy:
        Block 1: 201-point curve → detects fine-grained dip shapes
        Block 2: 100-point summary → detects coarser patterns
        Block 3: 50-point summary  → captures global context
      This is the 1D equivalent of image pooling in ResNets.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.conv  = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2   # 'same' padding: keeps sequence length constant
        )
        self.bn    = nn.BatchNorm1d(out_channels)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, seq_len)
        x = self.conv(x)               # → (batch, out_channels, seq_len)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)               # → (batch, out_channels, seq_len // 2)
        return x


class ExoplanetCNN(nn.Module):
    """
    Full 1D CNN for binary classification of Kepler light curves.

    ┌─────────────────────────────────────────────────────────────┐
    │  INPUT  (batch, 1, 201)  — 1 channel, 201 phase bins        │
    ├─────────────────────────────────────────────────────────────┤
    │  ConvBlock 1  1→16  ch, kernel=5  → (batch, 16, 100)        │
    │  ConvBlock 2  16→32 ch, kernel=5  → (batch, 32,  50)        │
    │  ConvBlock 3  32→64 ch, kernel=3  → (batch, 64,  25)        │
    ├─────────────────────────────────────────────────────────────┤
    │  Global Average Pooling             → (batch, 64)            │
    │  (replaces Flatten — more robust to variable-length inputs) │
    ├─────────────────────────────────────────────────────────────┤
    │  FC: 64 → 128, ReLU                                          │
    │  Dropout(0.5)                                                │
    │  FC: 128 → 2   (logits; no softmax — CrossEntropyLoss does) │
    └─────────────────────────────────────────────────────────────┘

    Why Global Average Pooling (GAP) instead of Flatten?
    ──────────────────────────────────────────────────────
    Flatten would give 64 × 25 = 1600 features → a massive FC layer prone to
    overfitting. GAP averages each of the 64 feature maps over the 25 remaining
    time steps, yielding just 64 numbers. This:
      ① Drastically reduces parameters (~90% fewer in the classifier head)
      ② Implicitly encourages each filter to be a global transit detector
      ③ Is invariant to small positional shifts of the transit peak

    Why Dropout(0.5)?
    ──────────────────
    During training, randomly zeroes 50% of FC activations each forward pass,
    forcing the network not to rely on any single neuron. Prevents co-adaptation
    and acts as an ensemble of 2^128 sub-networks at inference time.
    """

    def __init__(
        self,
        input_length: int = config.INPUT_LENGTH,
        num_classes:  int = config.NUM_CLASSES,
        dropout:      float = config.DROPOUT_RATE
    ):
        super().__init__()

        # ── Convolutional Feature Extractor ───────────────────────────────────
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels=1,  out_channels=16, kernel_size=5),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=5),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3),
        )

        # ── Global Average Pooling ─────────────────────────────────────────────
        # AdaptiveAvgPool1d(1) collapses the time dimension to length 1
        # regardless of input length → output is (batch, 64, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

        # ── Weight Initialisation ─────────────────────────────────────────────
        # Kaiming (He) init is optimal for ReLU networks.
        # It sets initial weights so variance is preserved through ReLU layers,
        # preventing vanishing/exploding gradients from the first epoch.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 1, 201)

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, 2)
            Raw (un-normalised) class scores.
            Pass through softmax to get probabilities.
        """
        # Feature extraction
        x = self.conv_blocks(x)    # → (batch, 64, ~25)

        # Global average pool → (batch, 64, 1)
        x = self.gap(x)

        # Flatten the last dim: (batch, 64, 1) → (batch, 64)
        x = x.squeeze(-1)

        # Classify
        logits = self.classifier(x)  # → (batch, 2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return softmax probabilities instead of logits."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Total trainable parameter count — useful to report in your README."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick sanity check


if __name__ == "__main__":
    model = ExoplanetCNN()
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    dummy = torch.randn(8, 1, config.INPUT_LENGTH)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # expect (8, 2)
    proba = model.predict_proba(dummy)
    print(f"Probabilities sum to 1: {proba.sum(dim=1)}")  # should all be 1.0