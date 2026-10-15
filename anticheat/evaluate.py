"""Evaluation, framed as intrusion detection rather than classification.

Three things this reports that a plain accuracy number hides:

  1. False-positive rate at a recall the operator would actually deploy at.
  2. Precision once cheaters are as rare in the population as they really are.
  3. Player-level performance after aggregating many windows, because banning on a
     single suspicious engagement is what makes a detector unusable in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import GROUP_COLS
from .metrics import (
    average_precision,
    confusion,
    fpr_at_recall,
    precision_at_prevalence,
    recall_at_fpr,
    roc_auc,
)
from .model import build_model
from .train import Standardiser, predict, prepare, resolve_device
from .windows import WindowSet


def aggregate_episodes(meta: pd.DataFrame, scores: np.ndarray, topk: int, alarm_threshold: float) -> pd.DataFrame:
    """Collapse one player's windows into a single score per agent-episode.

    AntiCheatPT names this as its own main limitation: one kill is not enough evidence
    to act on. Aggregation is where a 3% per-window false-positive rate either becomes
    tolerable or compounds into an alarm on nearly every honest player.
    """

    frame = meta.copy()
    frame["score"] = scores
    frame["alarm"] = (scores >= alarm_threshold).astype(float)

    def _topk_mean(s: pd.Series) -> float:
        return float(np.sort(s.to_numpy())[::-1][:topk].mean())

    grouped = frame.groupby(GROUP_COLS, sort=False)
    out = grouped.agg(
        y=("y", "max"),
        label_type=("label_type", "first"),
        windows=("score", "size"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        alarms=("alarm", "sum"),
    )
    out["topk_score"] = grouped["score"].apply(_topk_mean)
    out["alarm_rate"] = out["alarms"] / out["windows"]
    return out.reset_index()


def _operating_points(y: np.ndarray, scores: np.ndarray, cfg: Config) -> dict:
    points = {
        "roc_auc": roc_auc(y, scores),
        "average_precision": average_precision(y, scores),
        "at_recall": {},
        "at_fpr": {},
    }
    for target in cfg.eval.recall_targets:
        points["at_recall"][f"{target:.2f}"] = fpr_at_recall(y, scores, target)
    for target in cfg.eval.fpr_targets:
        points["at_fpr"][f"{target:g}"] = recall_at_fpr(y, scores, target)
    return points


def _prevalence_table(tpr: float, fpr: float, cfg: Config) -> dict:
    return {f"{p:g}": precision_at_prevalence(tpr, fpr, p) for p in cfg.eval.prevalences}


def _per_cheat_recall(meta: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    frame = meta.copy()
    frame["hit"] = (scores >= threshold).astype(float)
    cheating = frame[frame["y"] == 1]
    if cheating.empty:
        return {}
    return {
        str(kind): {"n": int(len(part)), "recall": float(part["hit"].mean())}
        for kind, part in cheating.groupby("label_type")
    }


def evaluate(run_dir: Path, split: str = "test", verbose: bool = True) -> dict:
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_json(run_dir / "config.json")

    _, windows, splits = prepare(cfg)

    if windows.features != checkpoint["features"]:
        raise RuntimeError(
            "the capture no longer produces the features this model was trained on; retrain"
        )

    scaler = Standardiser.from_state(checkpoint["scaler"])
    device = resolve_device(cfg.train.device)

    model = build_model(checkpoint["n_features"], cfg.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    def score(subset: WindowSet) -> np.ndarray:
        return predict(model, scaler(subset.X), device)

    val_set = windows.subset(splits["val"])
    test_set = windows.subset(splits[split])

    val_scores = score(val_set)
    test_scores = score(test_set)

    # The deployment threshold is chosen on validation, never on the split being
    # reported, or the false-positive rate quoted is the best case rather than the
    # expected one.
    target = cfg.eval.recall_targets[1] if len(cfg.eval.recall_targets) > 1 else cfg.eval.recall_targets[0]
    tuned = fpr_at_recall(val_set.y, val_scores, target)
    threshold = tuned["threshold"] if np.isfinite(tuned["threshold"]) else cfg.eval.threshold

    window_points = _operating_points(test_set.y, test_scores, cfg)
    window_conf = confusion(test_set.y, test_scores, threshold)

    episodes = aggregate_episodes(test_set.meta, test_scores, cfg.eval.topk, threshold)
    episode_report = {}
    for column in ("mean_score", "max_score", "topk_score", "alarm_rate"):
        values = episodes[column].to_numpy()
        episode_report[column] = _operating_points(episodes["y"].to_numpy(), values, cfg)

    report = {
        "split": split,
        "threshold": float(threshold),
        "threshold_from": f"val, recall>={target:.2f}",
        "n_windows": int(len(test_set)),
        "n_episodes": int(len(episodes)),
        "window_level": {
            **window_points,
            "at_threshold": window_conf,
            "precision_at_prevalence": _prevalence_table(
                window_conf["recall"], window_conf["fpr"], cfg
            ),
            "per_cheat_recall": _per_cheat_recall(test_set.meta, test_scores, threshold),
        },
        "episode_level": episode_report,
    }

    (run_dir / f"metrics_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    episodes.to_csv(run_dir / f"episode_scores_{split}.csv", index=False)

    if verbose:
        print(format_report(report))

    return report


def format_report(r: dict) -> str:
    w = r["window_level"]
    c = w["at_threshold"]

    lines = [
        f"=== {r['split']} split: {r['n_windows']} windows over {r['n_episodes']} agent-episodes ===",
        f"threshold {r['threshold']:.4f}  (chosen on {r['threshold_from']})",
        "",
        "-- window level --",
        f"ROC AUC {w['roc_auc']:.4f}   average precision {w['average_precision']:.4f}",
        f"accuracy {c['accuracy']:.4f}  precision {c['precision']:.4f}  recall {c['recall']:.4f}  "
        f"F1 {c['f1']:.4f}  specificity {c['specificity']:.4f}",
        f"confusion  tp {c['tp']:.0f}  fp {c['fp']:.0f}  tn {c['tn']:.0f}  fn {c['fn']:.0f}",
        "",
        "false-positive rate at fixed recall:",
    ]
    for target, point in w["at_recall"].items():
        lines.append(f"  recall >= {target}   fpr {point['fpr']:.4f}  (threshold {point['threshold']:.4f})")

    lines.append("")
    lines.append("recall at fixed false-positive rate:")
    for target, point in w["at_fpr"].items():
        lines.append(f"  fpr <= {target:>6}   recall {point['tpr']:.4f}  (threshold {point['threshold']:.4f})")

    lines.append("")
    lines.append("precision if cheaters are this rare in the population:")
    for prevalence, precision in w["precision_at_prevalence"].items():
        lines.append(f"  prevalence {prevalence:>5}   precision {precision:.4f}")

    if w["per_cheat_recall"]:
        lines.append("")
        lines.append("recall by cheat type at the operating threshold:")
        for kind, stat in sorted(w["per_cheat_recall"].items()):
            lines.append(f"  {kind:<12} n {stat['n']:>5}   recall {stat['recall']:.4f}")

    lines.append("")
    lines.append("-- agent-episode level (windows aggregated per player) --")
    lines.append(f"{'aggregator':<12} {'ROC AUC':>8} {'AP':>8}   fpr at recall 0.90")
    for name, points in r["episode_level"].items():
        at90 = points["at_recall"].get("0.90", {})
        fpr = at90.get("fpr", float("nan"))
        lines.append(
            f"{name:<12} {points['roc_auc']:>8.4f} {points['average_precision']:>8.4f}   {fpr:.4f}"
        )

    return "\n".join(lines)
