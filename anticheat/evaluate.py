"""Evaluation, framed as intrusion detection rather than classification.

Three things this reports that a plain accuracy number hides:

  1. False-positive rate at a recall the operator would actually deploy at.
  2. Precision once cheaters are as rare in the population as they really are.
  3. Player-level performance after aggregating many windows, because banning on a
     single suspicious engagement is what makes a detector unusable in production.

The "cheating" head is the alarm and gets the full treatment. The per-cheat heads are
reported alongside it, with recall broken down by severity where the cheat has one, so
the result reads "detects trigger-bot down to severity X" rather than one pooled number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import CHEAT_COLUMNS, GROUP_COLS, HEADS
from .metrics import (
    average_precision,
    confusion,
    fpr_at_recall,
    precision_at_prevalence,
    recall_at_fpr,
    roc_auc,
)
from .model import build_model
from .train import Standardiser, head_scores, predict, prepare, resolve_device
from .windows import CHEATING, EPISODE_MASK_COLS, EPISODE_TARGET_COLS, WindowSet

AGGREGATORS = ("mean", "max", "topk", "alarm_rate")

# Wallhack is binary by design (on or off), so only these two have a severity axis.
SEVERITY_HEADS = ("aimsnap", "triggerbot")
SEVERITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def aggregate_episodes(windows: WindowSet, scores: np.ndarray, thresholds: dict, topk: int) -> pd.DataFrame:
    """Collapse one player's windows into a score per head per agent-episode.

    AntiCheatPT names this as its own main limitation: one kill is not enough evidence
    to act on. Aggregation is where a 3% per-window false-positive rate either becomes
    tolerable or compounds into an alarm on nearly every honest player.

    Every window is pooled, masked or not, and the target is the episode's: a deployed
    detector cannot know which of a player's windows hold a firing, so it has to judge
    the player from all of them.
    """

    frame = windows.meta[GROUP_COLS + ["label_type", "label_aim_snap", "label_trigger_bot"]
                         + EPISODE_TARGET_COLS + EPISODE_MASK_COLS].copy()
    for i, head in enumerate(HEADS):
        frame[f"s_{head}"] = scores[:, i]
        frame[f"a_{head}"] = (scores[:, i] >= thresholds[head]).astype(float)

    spec = {
        "label_type": ("label_type", "first"),
        "label_aim_snap": ("label_aim_snap", "first"),
        "label_trigger_bot": ("label_trigger_bot", "first"),
        "windows": (f"s_{HEADS[0]}", "size"),
    }
    for head in HEADS:
        spec[f"y_{head}"] = (f"ey_{head}", "first")
        spec[f"m_{head}"] = (f"em_{head}", "first")
        spec[f"mean_{head}"] = (f"s_{head}", "mean")
        spec[f"max_{head}"] = (f"s_{head}", "max")
        spec[f"alarm_rate_{head}"] = (f"a_{head}", "mean")

    out = frame.groupby(GROUP_COLS, sort=False).agg(**spec)

    for head in HEADS:
        top = (
            frame.sort_values(f"s_{head}", ascending=False)
            .groupby(GROUP_COLS, sort=False)
            .head(topk)
            .groupby(GROUP_COLS)[f"s_{head}"]
            .mean()
        )
        out[f"topk_{head}"] = top.reindex(out.index)

    return out.reset_index()


def _operating_points(y: np.ndarray, scores: np.ndarray, cfg: Config) -> dict:
    points = {
        "roc_auc": roc_auc(y, scores),
        "average_precision": average_precision(y, scores),
        "at_recall": {},
        "at_fpr": {},
    }
    # roc_curve has no points to search on an empty split, and every lookup would fail.
    if len(y) == 0:
        return points

    for target in cfg.eval.recall_targets:
        points["at_recall"][f"{target:.2f}"] = fpr_at_recall(y, scores, target)
    for target in cfg.eval.fpr_targets:
        points["at_fpr"][f"{target:g}"] = recall_at_fpr(y, scores, target)
    return points


def _prevalence_table(tpr: float, fpr: float, cfg: Config) -> dict:
    return {f"{p:g}": precision_at_prevalence(tpr, fpr, p) for p in cfg.eval.prevalences}


def _recall_by_type(windows: WindowSet, scores: np.ndarray, threshold: float) -> dict:
    """The alarm's recall split by what the agent was actually running."""

    cheating = windows.y[:, CHEATING] == 1
    if not cheating.any():
        return {}

    frame = windows.meta.loc[cheating, ["label_type"]].copy()
    frame["hit"] = (scores[cheating, CHEATING] >= threshold).astype(float)
    return {
        str(kind): {"n": int(len(part)), "recall": float(part["hit"].mean())}
        for kind, part in frame.groupby("label_type")
    }


def _recall_by_severity(windows: WindowSet, scores: np.ndarray, head: str, threshold: float) -> dict:
    i = HEADS.index(head)
    severity_column = CHEAT_COLUMNS[head][0]

    positive = (windows.y[:, i] == 1) & (windows.mask[:, i] > 0)
    severity = windows.meta.loc[positive, severity_column].to_numpy()
    hit = scores[positive, i] >= threshold

    out = {}
    for lo, hi in zip(SEVERITY_BINS[:-1], SEVERITY_BINS[1:]):
        in_bin = (severity > lo) & (severity <= hi)
        if in_bin.any():
            out[f"{lo:.1f}-{hi:.1f}"] = {"n": int(in_bin.sum()), "recall": float(hit[in_bin].mean())}
    return out


def evaluate(run_dir: Path, split: str = "test", verbose: bool = True) -> dict:
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_json(run_dir / "config.json")

    if list(checkpoint.get("heads", [])) != list(HEADS):
        raise RuntimeError("this checkpoint predates the current heads; retrain")

    _, windows, splits = prepare(cfg)

    if windows.features != checkpoint["features"]:
        raise RuntimeError(
            "the capture no longer produces the features this model was trained on; retrain"
        )

    scaler = Standardiser.from_state(checkpoint["scaler"])
    device = resolve_device(cfg.train.device)

    model = build_model(checkpoint["n_features"], cfg.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    val_set = windows.subset(splits["val"], with_x=False)
    test_set = windows.subset(splits[split], with_x=False)

    val_scores = predict(model, windows, splits["val"], scaler, device)
    test_scores = predict(model, windows, splits[split], scaler, device)

    # Each head's deployment threshold is chosen on validation, never on the split being
    # reported, or the false-positive rate quoted is the best case rather than the
    # expected one.
    target = cfg.eval.recall_targets[1] if len(cfg.eval.recall_targets) > 1 else cfg.eval.recall_targets[0]

    thresholds = {}
    heads_report = {}

    for i, head in enumerate(HEADS):
        yv, sv = head_scores(val_set, val_scores, i)
        yt, st = head_scores(test_set, test_scores, i)

        tuned = fpr_at_recall(yv, sv, target) if len(yv) else {"threshold": float("nan")}
        threshold = tuned["threshold"] if np.isfinite(tuned["threshold"]) else cfg.eval.threshold
        thresholds[head] = float(threshold)

        conf = confusion(yt, st, threshold)
        entry = {
            "threshold": float(threshold),
            "n_windows": int(len(yt)),
            "positives": int(yt.sum()),
            **_operating_points(yt, st, cfg),
            "at_threshold": conf,
            "precision_at_prevalence": _prevalence_table(conf["recall"], conf["fpr"], cfg),
        }
        if head in SEVERITY_HEADS:
            entry["recall_by_severity"] = _recall_by_severity(test_set, test_scores, head, threshold)
        heads_report[head] = entry

    heads_report["cheating"]["recall_by_type"] = _recall_by_type(test_set, test_scores, thresholds["cheating"])

    episodes = aggregate_episodes(test_set, test_scores, thresholds, cfg.eval.topk)
    episode_report = {}
    for head in HEADS:
        keep = episodes[f"m_{head}"] > 0
        y = episodes.loc[keep, f"y_{head}"].to_numpy()
        episode_report[head] = {
            agg: _operating_points(y, episodes.loc[keep, f"{agg}_{head}"].to_numpy(), cfg)
            for agg in AGGREGATORS
        }

    report = {
        "split": split,
        "threshold_from": f"val, recall>={target:.2f}",
        "n_windows": int(len(test_set)),
        "n_episodes": int(len(episodes)),
        "window_level": heads_report,
        "episode_level": episode_report,
    }

    (run_dir / f"metrics_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    episodes.to_csv(run_dir / f"episode_scores_{split}.csv", index=False)

    if verbose:
        print(format_report(report))

    return report


def format_report(r: dict) -> str:
    alarm = r["window_level"]["cheating"]
    c = alarm["at_threshold"]

    lines = [
        f"=== {r['split']} split: {r['n_windows']} windows over {r['n_episodes']} agent-episodes ===",
        f"thresholds chosen on {r['threshold_from']}, per head",
        "",
        "-- cheating head (the alarm), window level --",
        f"threshold {alarm['threshold']:.4f}",
        f"ROC AUC {alarm['roc_auc']:.4f}   average precision {alarm['average_precision']:.4f}",
        f"accuracy {c['accuracy']:.4f}  precision {c['precision']:.4f}  recall {c['recall']:.4f}  "
        f"F1 {c['f1']:.4f}  specificity {c['specificity']:.4f}",
        f"confusion  tp {c['tp']:.0f}  fp {c['fp']:.0f}  tn {c['tn']:.0f}  fn {c['fn']:.0f}",
        "",
        "false-positive rate at fixed recall:",
    ]
    for target, point in alarm["at_recall"].items():
        lines.append(f"  recall >= {target}   fpr {point['fpr']:.4f}  (threshold {point['threshold']:.4f})")

    lines.append("")
    lines.append("recall at fixed false-positive rate:")
    for target, point in alarm["at_fpr"].items():
        lines.append(f"  fpr <= {target:>6}   recall {point['tpr']:.4f}  (threshold {point['threshold']:.4f})")

    lines.append("")
    lines.append("precision if cheaters are this rare in the population:")
    for prevalence, precision in alarm["precision_at_prevalence"].items():
        lines.append(f"  prevalence {prevalence:>5}   precision {precision:.4f}")

    if alarm.get("recall_by_type"):
        lines.append("")
        lines.append("alarm recall by what the agent was running:")
        for kind, stat in sorted(alarm["recall_by_type"].items()):
            lines.append(f"  {kind:<12} n {stat['n']:>6}   recall {stat['recall']:.4f}")

    lines.append("")
    lines.append("-- every head, window level --")
    lines.append(f"{'head':<12} {'windows':>8} {'pos':>7} {'ROC AUC':>8} {'AP':>8} {'recall':>8} {'fpr':>8}")
    for head, h in r["window_level"].items():
        t = h["at_threshold"]
        lines.append(f"{head:<12} {h['n_windows']:>8} {h['positives']:>7} {h['roc_auc']:>8.4f} "
                     f"{h['average_precision']:>8.4f} {t['recall']:>8.4f} {t['fpr']:>8.4f}")

    for head in SEVERITY_HEADS:
        bins = r["window_level"][head].get("recall_by_severity", {})
        if not bins:
            continue
        lines.append("")
        lines.append(f"{head} recall by severity:")
        for span, stat in bins.items():
            lines.append(f"  {span}   n {stat['n']:>6}   recall {stat['recall']:.4f}")

    lines.append("")
    lines.append("-- agent-episode level (windows aggregated per player) --")
    lines.append(f"{'head':<12} {'aggregator':<12} {'ROC AUC':>8} {'AP':>8}   fpr at recall 0.90")
    for head, by_agg in r["episode_level"].items():
        # The alarm gets every aggregator; the per-cheat heads just the top-k mean.
        for agg in (AGGREGATORS if head == "cheating" else ("topk",)):
            points = by_agg[agg]
            fpr = points["at_recall"].get("0.90", {}).get("fpr", float("nan"))
            lines.append(f"{head:<12} {agg:<12} {points['roc_auc']:>8.4f} "
                         f"{points['average_precision']:>8.4f}   {fpr:.4f}")

    return "\n".join(lines)
