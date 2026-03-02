# =============================================================================
# src/train.py — Full Training Pipeline
#
# What happens here:
#   1. Load the preprocessed dataset (or build it from scratch)
#   2. Split into train / val / test with stratification
#   3. Handle class imbalance with a weighted sampler
#   4. Instantiate model, loss, optimizer, and LR scheduler
#   5. Run the training loop with validation at every epoch
#   6. Save the best checkpoint based on validation AUC
#   7. Evaluate final performance on the held-out test set
# =============================================================================

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import build_dataset
from src.model import ExoplanetCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """
    Create a sampler that up-samples the minority class (CONFIRMED planets).

    The KOI dataset is imbalanced: there are ~3× more FALSE POSITIVEs than
    CONFIRMED planets. If we train naively, the model will learn to predict
    FALSE POSITIVE almost always and still achieve high accuracy.

    WeightedRandomSampler solves this by assigning each sample a weight
    inversely proportional to its class frequency, so BOTH classes appear
    equally often in each batch — without throwing away any data.

    This is superior to:
      ① Oversampling (duplicating minority samples → overfitting)
      ② Undersampling (discarding majority samples → loss of information)
      ③ class_weight in the loss (less stable with small minority classes)
    """
    class_counts = np.bincount(y_train)           # [count_FP, count_CONFIRMED]
    class_weights = 1.0 / class_counts            # inverse frequency
    sample_weights = class_weights[y_train]       # per-sample weight

    return WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def make_dataloaders(X: np.ndarray, y: np.ndarray):
    """
    Split data → train/val/test → wrap in PyTorch DataLoaders.

    Split strategy
    ──────────────
    We do two successive stratified splits:
      ① train+val vs test  (85% / 15%)
      ② train vs val       (from the 85%: 82% / 18% ≈ 70% / 15% globally)

    Stratified = each split preserves the original class ratio.
    This is critical: a test set with zero minority-class samples would give
    meaningless evaluation metrics.

    Returns train_loader, val_loader, test_loader
    """
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # ── Split ──────────────────────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT,
        stratify=y,
        random_state=config.RANDOM_SEED
    )

    val_fraction_of_trainval = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=config.RANDOM_SEED
    )

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── TensorDatasets ────────────────────────────────────────────────────────
    def to_tensors(X_, y_):
        return TensorDataset(torch.FloatTensor(X_), torch.LongTensor(y_))

    train_ds = to_tensors(X_train, y_train)
    val_ds   = to_tensors(X_val,   y_val)
    test_ds  = to_tensors(X_test,  y_test)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sampler = make_weighted_sampler(y_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,         # WeightedRandomSampler replaces shuffle=True
        num_workers=0,           # set > 0 on Linux for faster loading
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,           # no shuffling needed for evaluation
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, y_test


# ─────────────────────────────────────────────────────────────────────────────
# One Epoch — Train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Single training epoch.

    The training loop follows the standard PyTorch pattern:
      1. Zero gradients (accumulated from previous batch — must clear them)
      2. Forward pass  (compute predictions)
      3. Compute loss  (measure how wrong we are)
      4. Backward pass (compute ∂loss/∂weight for every parameter via autograd)
      5. Gradient clip (prevents exploding gradients from destabilising training)
      6. Optimizer step (update weights: w ← w - lr × ∇w)

    Returns average loss and accuracy over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # ① Zero gradients — PyTorch accumulates them by default
        optimizer.zero_grad()

        # ② Forward pass
        logits = model(X_batch)

        # ③ Cross-Entropy loss
        # = -log(softmax(logits)[correct_class])
        # Combines LogSoftmax + NLLLoss for numerical stability
        loss = criterion(logits, y_batch)

        # ④ Backward pass — fills .grad fields via autograd chain rule
        loss.backward()

        # ⑤ Gradient clipping: clip gradient norm to 1.0
        # Prevents a single bad batch from causing a catastrophic weight update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ⑥ Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# One Epoch — Validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, float]:
    """
    Evaluate model on val or test set.

    @torch.no_grad() disables gradient computation entirely:
      - No computation graph is built → ~50% less memory
      - Faster inference (no tracking of intermediate activations)

    Returns loss, accuracy, and AUC-ROC.

    Why AUC-ROC instead of accuracy alone?
    ────────────────────────────────────────
    Accuracy is misleading on imbalanced classes. AUC-ROC measures the model's
    ability to rank CONFIRMED above FALSE POSITIVE regardless of threshold.
    AUC=0.5 → random; AUC=1.0 → perfect separation.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        probs  = torch.softmax(logits, dim=1)[:, 1]  # P(CONFIRMED)

        total_loss  += loss.item() * len(y_batch)
        preds        = logits.argmax(dim=1)
        correct      += (preds == y_batch).sum().item()
        total        += len(y_batch)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    auc      = roc_auc_score(
        np.concatenate(all_labels),
        np.concatenate(all_probs)
    )

    return avg_loss, accuracy, auc


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(force_rebuild: bool = False):
    """
    Full training run. Call this from __main__ or from a notebook.

    Learning Rate Scheduling
    ─────────────────────────
    We use ReduceLROnPlateau: if validation AUC doesn't improve for `patience`
    epochs, the LR is multiplied by `factor` (e.g., 0.5). This lets us start
    with a large LR for fast initial convergence, then fine-tune with a small LR.

    Early Stopping
    ───────────────
    We save the checkpoint whenever val AUC improves. If AUC hasn't improved
    for `patience` epochs, training stops early — no need to specify epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = build_dataset(force_rebuild=force_rebuild)
    train_loader, val_loader, test_loader, y_test = make_dataloaders(X, y)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ExoplanetCNN().to(device)
    log.info(f"Model parameters: {model.count_parameters():,}")


    # ── Loss: CrossEntropyLoss ────────────────────────────────────────────────
    # WeightedRandomSampler already handles class imbalance in batches.
    # We keep the loss simple — threshold tuning handles precision/recall trade-off.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer: AdamW ──────────────────────────────────────────────────────
    # AdamW = Adam with decoupled weight decay (L2 regularisation applied to
    # weights directly, NOT through the gradient — more theoretically correct).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",       # we want to MAXimise val_auc
        factor=0.5,       # multiply LR by 0.5 on plateau
        patience=5
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    patience_cnt  = 0
    early_stop    = 10   # stop after 10 epochs of no improvement
    history       = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        log.info(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  AUC: {val_auc:.4f}"
        )

        # ── Checkpoint best model ──────────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                },
                config.MODEL_PATH
            )
            log.info(f"  ✓ New best AUC: {val_auc:.4f} — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {early_stop} epochs)")
                break

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    log.info("\n─── TEST SET EVALUATION ───")
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    log.info(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")

    # Detailed report — using same threshold as predict.py for consistency
    PLANET_THRESHOLD = 0.68  # tuned for ~82% precision on confirmed planets

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs >= PLANET_THRESHOLD).astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    print(f"\nClassification Report (threshold={PLANET_THRESHOLD}):")
    print(classification_report(
        all_labels, all_preds,
        target_names=["False Positive", "Confirmed Planet"]
    ))

    # ── Plot Training History ─────────────────────────────────────────────────
    _plot_history(history)
    _plot_confusion_matrix(all_labels, all_preds)

    return model, history



def _plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_auc"],  label="Val AUC",      color="green")
    axes[1].plot(history["val_acc"],  label="Val Accuracy",  color="orange")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(config.LOG_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    log.info(f"Training history saved to {path}")
    plt.close()


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["False Positive", "Confirmed"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"Confusion matrix saved to {path}")
    plt.close()


if __name__ == "__main__":
    train()