"""Training-loop helpers shared by the baseline workflows."""

from __future__ import annotations

import copy
import math


def select_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BestEpochTracker:
    """Keep the weights of the best validation epoch and signal early stopping.

    ``patience=0`` disables early stopping but still restores the best epoch.
    NaN scores (e.g. ROC-AUC on a single-class split) never count as improvements.
    """

    def __init__(self, patience: int = 0, higher_is_better: bool = True) -> None:
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score = -math.inf if higher_is_better else math.inf
        self.best_epoch = 0
        self.best_state = None
        self.epochs_without_improvement = 0

    def update(self, epoch: int, score: float, model) -> bool:
        improved = not math.isnan(score) and (
            score > self.best_score if self.higher_is_better else score < self.best_score
        )
        if improved or self.best_state is None:
            if improved:
                self.best_score = score
            self.best_epoch = epoch
            self.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.epochs_without_improvement >= self.patience

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(copy.deepcopy(self.best_state))

    def summary(self) -> dict[str, float]:
        return {"best_epoch": self.best_epoch, "best_val_score": self.best_score}
