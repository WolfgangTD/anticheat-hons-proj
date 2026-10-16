"""Training loop.

Class imbalance is handled by weighting the loss rather than by duplicating minority
windows with added noise. Augmented copies of a window are near-identical to their
source, and once they exist the split has to keep every copy on the same side of the
fold boundary or the score is fiction.

Folds are never copied out of the window array. Batches are gathered by index and
standardised on the device, so a multi-hour capture costs one copy of its windows, and
with the cache that copy is a memory-mapped file rather than process memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import load_or_build
from .config import Config
from .data import HEADS
from .metrics import average_precision, roc_auc
from .model import build_model, count_parameters
from .windows import CHEATING, WindowSet, describe_split, group_split


class Standardiser:
    """Per-feature z-scoring, fitted on the training fold only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @staticmethod
    def fit(X: np.ndarray, idx: np.ndarray, chunk: int = 8192) -> "Standardiser":
        """Accumulated in chunks, so the training fold is never materialised whole."""

        n_features = X.shape[2]
        total = np.zeros(n_features)
        total_sq = np.zeros(n_features)
        count = 0

        idx = np.sort(idx)
        for i in range(0, len(idx), chunk):
            block = np.asarray(X[idx[i : i + chunk]], dtype=np.float64).reshape(-1, n_features)
            total += block.sum(axis=0)
            total_sq += np.square(block).sum(axis=0)
            count += len(block)

        mean = total / max(count, 1)
        std = np.sqrt(np.maximum(total_sq / max(count, 1) - np.square(mean), 0.0))
        # A constant feature carries no information; dividing by its zero spread would
        # turn it into inf and take the whole run with it.
        std = np.where(std < 1e-6, 1.0, std)
        return Standardiser(mean.astype(np.float32), std.astype(np.float32))

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.mean).to(device), torch.from_numpy(self.std).to(device)

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


def _batch(X: np.ndarray, idx: np.ndarray, device: torch.device, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # Sorted indices read a memory-mapped array front to back instead of seeking around it.
    xb = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)]))
    return (xb.to(device, non_blocking=True) - mean) / std


@torch.no_grad()
def _score(model: nn.Module, windows: WindowSet, idx: np.ndarray, scaler: Standardiser,
           device: torch.device, criterion: nn.Module | None = None, batch_size: int = 1024):
    """Sigmoid scores for windows `idx` in sorted order, plus per-head mean loss if asked."""

    model.eval()
    mean, std = scaler.tensors(device)
    idx = np.sort(idx)

    scores = []
    loss_sum = torch.zeros(len(HEADS), device=device)
    loss_n = torch.zeros(len(HEADS), device=device)

    for i in range(0, len(idx), batch_size):
        part = idx[i : i + batch_size]
        logits = model(_batch(windows.X, part, device, mean, std))
        scores.append(torch.sigmoid(logits).cpu().numpy())

        if criterion is not None:
            yb = torch.from_numpy(windows.y[part]).to(device)
            mb = torch.from_numpy(windows.mask[part]).to(device)
            loss_sum += (criterion(logits, yb) * mb).sum(dim=0)
            loss_n += mb.sum(dim=0)

    out = np.concatenate(scores) if scores else np.zeros((0, len(HEADS)), dtype=np.float32)
    if criterion is None:
        return out
    return out, (loss_sum / loss_n.clamp(min=1.0)).cpu().numpy()


def predict(model: nn.Module, windows: WindowSet, idx: np.ndarray, scaler: Standardiser,
            device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """Sigmoid scores, (len(idx), H) in HEADS order, for windows `idx` in sorted order."""

    return _score(model, windows, idx, scaler, device, batch_size=batch_size)


def head_scores(windows: WindowSet, scores: np.ndarray, head: int) -> tuple[np.ndarray, np.ndarray]:
    """One head's targets and scores, with the windows it has no valid target for removed."""

    keep = windows.mask[:, head] > 0
    return windows.y[keep, head], scores[keep, head]


def masked_pos_weight(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """n_neg / n_pos per head, counted only over the windows each head trains on."""

    pos = (y * mask).sum(axis=0)
    neg = ((1.0 - y) * mask).sum(axis=0)
    return (neg / np.maximum(pos, 1.0)).astype(np.float32)


def prepare(cfg: Config) -> tuple[str, WindowSet, dict[str, np.ndarray]]:
    summary, windows = load_or_build(cfg)
    return summary, windows, group_split(windows, cfg.split)


def _comparable(config: dict) -> dict:
    # Only patience may change on resume. Not the epoch count: the cosine schedule is
    # laid out over it, and the saved scheduler would keep decaying on the old length.
    train = {k: v for k, v in config["train"].items() if k != "patience"}
    return {**config, "train": train}


def train(cfg: Config, out_dir: Path, verbose: bool = True, resume: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.train.seed)

    summary, windows, splits = prepare(cfg)

    if verbose:
        print(summary)
        print()
        print(f"windows           {len(windows)} x {windows.X.shape[1]} steps x {windows.X.shape[2]} features")
        print()
        print(describe_split(windows, splits))
        print()

    train_idx, val_idx = splits["train"], splits["val"]
    val_set = windows.subset(val_idx, with_x=False)

    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(windows.X.shape[2], cfg.model).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)

    history: list[dict] = []
    best = {"ap": -1.0, "epoch": -1}
    best_state = None
    first_epoch = 1
    last_path = out_dir / "last.pt"

    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if _comparable(state["config"]) != _comparable(asdict(cfg)) or state["features"] != windows.features:
            raise RuntimeError(f"{last_path} was trained with different settings or data; use a new --out")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler = Standardiser.from_state(state["scaler"])
        pos_weight = np.asarray(state["pos_weight"], dtype=np.float32)
        history, best, best_state = state["history"], state["best"], state["best_state"]
        first_epoch = state["epoch"] + 1
        if verbose:
            print(f"resumed from {last_path} after epoch {state['epoch']}")
    else:
        scaler = Standardiser.fit(windows.X, train_idx)
        if cfg.train.pos_weight == "auto":
            pos_weight = masked_pos_weight(windows.y[train_idx], windows.mask[train_idx])
        else:
            pos_weight = np.full(len(HEADS), float(cfg.train.pos_weight), dtype=np.float32)

    # Per-element loss, so the mask can drop a head's term for a window without dropping
    # the window from the other heads.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device), reduction="none")
    mean, std = scaler.tensors(device)

    if verbose:
        weights = "  ".join(f"{h} {w:.2f}" for h, w in zip(HEADS, pos_weight))
        print(f"LSTM + transformer encoder on {device}, {count_parameters(model):,} parameters")
        print(f"pos_weight  {weights}")
        print()

    started = time.time()
    stopped_early = bool(history) and history[-1]["epoch"] - best["epoch"] >= cfg.train.patience

    for epoch in range(first_epoch, cfg.train.epochs + 1):
        if stopped_early:
            break

        epoch_started = time.time()
        model.train()
        loss_sum = torch.zeros(len(HEADS), device=device)
        loss_n = torch.zeros(len(HEADS), device=device)

        # Seeded per epoch, so a resumed run shuffles exactly as an uninterrupted one would.
        order = np.random.default_rng(cfg.train.seed + epoch).permutation(train_idx)
        lr = optimiser.param_groups[0]["lr"]

        for i in range(0, len(order), cfg.train.batch_size):
            part = np.sort(order[i : i + cfg.train.batch_size])
            xb = _batch(windows.X, part, device, mean, std)
            yb = torch.from_numpy(windows.y[part]).to(device)
            mb = torch.from_numpy(windows.mask[part]).to(device)

            optimiser.zero_grad(set_to_none=True)
            terms = criterion(model(xb), yb) * mb
            loss = terms.sum() / mb.sum().clamp(min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()

            loss_sum += terms.detach().sum(dim=0)
            loss_n += mb.sum(dim=0)

        scheduler.step()

        train_loss = (loss_sum / loss_n.clamp(min=1.0)).cpu().numpy()
        scores, val_loss = _score(model, windows, val_idx, scaler, device, criterion)

        # Selection is on average precision, not AUC: with rare positives, AUC stays
        # comfortable while precision collapses, and precision is what gets an
        # innocent player banned. Averaged over the heads, so the checkpoint kept is
        # not one that bought a better cheating head by giving up the trigger-bot one.
        per_head = {head: average_precision(*head_scores(val_set, scores, i)) for i, head in enumerate(HEADS)}
        ap = float(np.nanmean(list(per_head.values())))
        auc = roc_auc(*head_scores(val_set, scores, CHEATING))
        seconds = time.time() - epoch_started

        history.append({
            "epoch": epoch,
            "lr": lr,
            "seconds": seconds,
            "train_loss": float((loss_sum.sum() / loss_n.sum().clamp(min=1.0)).item()),
            "train_loss_by_head": dict(zip(HEADS, train_loss.tolist())),
            "val_loss_by_head": dict(zip(HEADS, val_loss.tolist())),
            "val_mean_ap": ap,
            "val_cheating_auc": auc,
            "val_ap": per_head,
        })

        if verbose:
            heads = "  ".join(f"{h} {v:.3f}" for h, v in per_head.items())
            print(f"epoch {epoch:>3}  {seconds:6.1f}s  lr {lr:.2e}  train loss {history[-1]['train_loss']:.4f}  "
                  f"val loss {float(np.mean(val_loss)):.4f}  val mean AP {ap:.4f}  ({heads})")

        if ap > best["ap"]:
            best = {"ap": ap, "auc": auc, "epoch": epoch, "per_head_ap": per_head}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best["epoch"] >= cfg.train.patience:
            stopped_early = True
            if verbose:
                print(f"early stop: no gain in {cfg.train.patience} epochs")

        # Every epoch, so a job killed by a queue time limit loses at most one epoch.
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state(),
            "pos_weight": pos_weight.tolist(),
            "history": history,
            "best": best,
            "best_state": best_state,
            "config": asdict(cfg),
            "features": windows.features,
        }, last_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "scaler": scaler.state(),
        "features": windows.features,
        "n_features": windows.X.shape[2],
        "heads": list(HEADS),
        "config": asdict(cfg),
        "best": best,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    cfg.to_json(out_dir / "config.json")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if verbose:
        print()
        print(f"best epoch {best['epoch']} (val mean AP {best['ap']:.4f}) in {time.time() - started:.1f}s")
        print(f"saved to {out_dir}")

    return {"best": best, "history": history}
