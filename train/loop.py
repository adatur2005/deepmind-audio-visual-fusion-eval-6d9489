"""The training loop.

Two decisions here carry more weight than the architecture: which checkpoint
gets selected, and what the loss is weighted by. Both are places where a
reasonable-looking choice quietly invalidates the evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    patience: int = 6
    seed: int = 0


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(min(range(len(self.val_loss)), key=lambda i: self.val_loss[i]))


def positive_weight(labels: torch.Tensor) -> torch.Tensor:
    """Weight for the positive class in BCE, from the training labels only.

    Computed from TRAINING labels, never from the full dataset. A weight
    derived from all clips encodes the test set's class balance into the
    training objective — a small leak, and one that is hard to see because it
    happens in a line that looks like routine bookkeeping.

    The ratio matters here because a generator-level split does not preserve
    class balance. Holding out a generator removes a block of synthetic clips,
    so the training set skews real, and an unweighted loss would push the
    model toward predicting 'real' by default.
    """
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())

    if positives == 0:
        raise ValueError(
            "training labels contain no positive examples — the split has "
            "removed an entire class"
        )

    return torch.tensor(negatives / positives)


def train(model: nn.Module, train_batches, val_batches,
          config: TrainConfig) -> tuple[nn.Module, TrainHistory]:
    """Train with early stopping on validation loss.

    Selection is on VALIDATION loss, and validation comes from the training
    generators. That is optimistic, as step 1 established, and it is the
    correct tradeoff: selecting on the held-out generators would tune the
    model against the test distribution across every epoch of every run.

    Early stopping is itself a form of hyperparameter selection. Choosing the
    epoch with the lowest test loss would be indefensible; choosing it on
    validation is standard practice precisely because validation is the set we
    have already accepted as compromised.
    """
    torch.manual_seed(config.seed)

    all_train_labels = torch.cat([labels for _, _, labels in train_batches])
    pos_weight = positive_weight(all_train_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = TrainHistory()

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for video, audio, labels in train_batches:
            optimizer.zero_grad()
            loss = criterion(model(video, audio), labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        history.train_loss.append(epoch_loss / max(1, len(train_batches)))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for video, audio, labels in val_batches:
                val_loss += float(criterion(model(video, audio), labels.float()))
        val_loss /= max(1, len(val_batches))
        history.val_loss.append(val_loss)

        # Snapshot on improvement rather than keeping only the final weights.
        #
        # Without this the model returned is whatever the last epoch produced,
        # which after `patience` epochs of no improvement is by definition a
        # worse model than the one at the best epoch.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, history
