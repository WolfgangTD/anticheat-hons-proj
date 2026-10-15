"""Training loop.

Class imbalance is handled by weighting the loss rather than by duplicating minority
windows with added noise. Augmented copies of a window are near-identical to their
source, and once they exist the split has to keep every copy on the same side of the
fold boundary or the score is fiction.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import Capture, load_capture, summarise
from .metrics import average_precision, roc_auc
from .model import build_model, count_parameters
from .windows import WindowSet, build_windows, describe_split, group_split


class Standardiser:
    """Per-feature z-scoring, fitted on the training fold only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @staticmethod
    def fit(X: np.ndarray) -> "Standardiser":
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
        # A constant feature carries no information; dividing by its zero spread would
        # turn it into inf and take the whole run with it.
        std = np.where(std < 1e-6, 1.0, std)
        return Standardiser(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_state(state: dict) -> "Standardiser":
        return Standardiser(
            np.asarray(state["mean"], dtype=np.float32),
            np.asarray(state["std"], dtype=np.float32),
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def prepare(cfg: Config) -> tuple[Capture, WindowSet, dict[str, np.ndarray]]:
    capture = load_capture(cfg.data)
    windows = build_windows(capture, cfg.data, cfg.window)
    splits = group_split(windows, cfg.split)
    return capture, windows, splits


def train(cfg: Config, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.train.seed)

    capture, windows, splits = prepare(cfg)

    if verbose:
        print(summarise(capture))
        print()
        print(f"windows           {len(windows)} x {windows.X.shape[1]} steps x {windows.X.shape[2]} features")
        print()
        print(describe_split(windows, splits))
        print()

    train_set = windows.subset(splits["train"])
    val_set = windows.subset(splits["val"])

    scaler = Standardiser.fit(train_set.X)
    Xtr, Xva = scaler(train_set.X), scaler(val_set.X)

    device = resolve_device(cfg.train.device)
    model = build_model(windows.X.shape[2], cfg.model).to(device)

    n_pos = float(train_set.y.sum())
    n_neg = float(len(train_set.y) - n_pos)
    pos_weight = (n_neg / max(n_pos, 1.0)) if cfg.train.pos_weight == "auto" else float(cfg.train.pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)

    if verbose:
        print(f"{cfg.model.arch} on {device}, {count_parameters(model):,} parameters, pos_weight {pos_weight:.2f}")
        print()

    loader = _loader(Xtr, train_set.y, cfg.train.batch_size, True, cfg.train.num_workers)

    history = []
    best = {"ap": -1.0, "epoch": -1}
    best_state = None
    started = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()
            total += float(loss) * len(xb)

        scheduler.step()

        scores = predict(model, Xva, device)
        # Selection is on average precision, not AUC: with rare positives, AUC stays
        # comfortable while precision collapses, and precision is what gets an
        # innocent player banned.
        ap = average_precision(val_set.y, scores)
        auc = roc_auc(val_set.y, scores)

        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_ap": ap, "val_auc": auc})

        if verbose:
            print(f"epoch {epoch:>3}  loss {total / len(Xtr):.4f}  val AP {ap:.4f}  val AUC {auc:.4f}")

        if ap > best["ap"]:
            best = {"ap": ap, "auc": auc, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best["epoch"] >= cfg.train.patience:
            if verbose:
                print(f"early stop: no gain in {cfg.train.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "scaler": scaler.state(),
        "features": windows.features,
        "n_features": windows.X.shape[2],
        "config": asdict(cfg),
        "best": best,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    cfg.to_json(out_dir / "config.json")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if verbose:
        print()
        print(f"best epoch {best['epoch']} (val AP {best['ap']:.4f}) in {time.time() - started:.1f}s")
        print(f"saved to {out_dir}")

    return {"best": best, "history": history}
