"""Event-anchored context windows and leak-free splits.

Sliding a window over every step spends most of its capacity on agents walking around
an empty arena. Cheats only express themselves in engagements, so windows are anchored
on the events where an honest player and a cheating one must differ: the shot, the
moment line of sight opens, the moment damage lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig, SplitConfig, WindowConfig
from .data import FOLD_COLS, GROUP_COLS, Capture, feature_columns, label_table

META_COLS = GROUP_COLS + ["anchor", "anchor_kind", "y", "label_type",
                          "label_wallhack", "label_aim_snap", "label_trigger_bot"]


@dataclass
class WindowSet:
    X: np.ndarray            # (N, T, F) float32
    y: np.ndarray            # (N,) float32
    meta: pd.DataFrame       # one row per window, carries the fold keys
    features: list[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(
            X=self.X[idx],
            y=self.y[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            features=self.features,
        )


def _anchor_indices(group: pd.DataFrame, cfg: WindowConfig) -> list[tuple[int, str]]:
    """Positions within one agent-episode that are worth centring a window on."""

    found: list[tuple[int, str]] = []

    for kind in cfg.anchors:
        if kind == "shot":
            hits = np.flatnonzero(group["shot"].to_numpy() == 1)
        elif kind == "damage":
            hits = np.flatnonzero(group["damage_dealt"].to_numpy() > 0)
        elif kind == "los_acquire":
            los = group["has_los"].to_numpy()
            hits = np.flatnonzero((los == 1) & (np.r_[0, los[:-1]] == 0))
        elif kind == "uniform":
            hits = np.arange(0, len(group), cfg.stride)
        else:
            raise ValueError(f"unknown anchor {kind!r}")

        found.extend((int(i), kind) for i in hits)

    if not found:
        return []

    # Overlapping windows are near-duplicates of each other, which inflates the
    # apparent dataset size and lets an easy episode dominate the loss.
    found.sort()
    kept = [found[0]]
    for idx, kind in found[1:]:
        if idx - kept[-1][0] >= cfg.min_gap:
            kept.append((idx, kind))
    return kept


def build_windows(capture: Capture, data_cfg: DataConfig, cfg: WindowConfig) -> WindowSet:
    features = feature_columns(data_cfg)
    labels = label_table(capture.episodes)
    pad_col = features.index("is_pad")

    blocks: list[np.ndarray] = []
    rows: list[dict] = []

    for key, group in capture.steps.groupby(GROUP_COLS, sort=False):
        anchors = _anchor_indices(group, cfg)
        if not anchors:
            continue

        arr = group[features].to_numpy(dtype=np.float32)
        n = len(arr)

        try:
            label = labels.loc[key]
        except KeyError:
            continue

        for idx, kind in anchors:
            lo = idx - cfg.pre + 1
            hi = idx + cfg.post + 1
            take = np.arange(lo, hi)
            out_of_range = (take < 0) | (take >= n)

            # Near an episode boundary the window is completed by repeating the edge
            # frame, and the padding is flagged so the model can discount it rather
            # than reading a fabricated freeze as behaviour.
            window = arr[np.clip(take, 0, n - 1)].copy()
            window[out_of_range, pad_col] = 1.0

            blocks.append(window)
            rows.append(
                {
                    "session": key[0],
                    "episode_id": key[1],
                    "agent_id": key[2],
                    "anchor": idx,
                    "anchor_kind": kind,
                    "y": float(label[data_cfg.label_column]),
                    "label_type": label["label_type"],
                    "label_wallhack": float(label["label_wallhack"]),
                    "label_aim_snap": float(label["label_aim_snap"]),
                    "label_trigger_bot": float(label["label_trigger_bot"]),
                }
            )

    if not blocks:
        raise RuntimeError("no windows produced; check the anchor settings")

    X = np.stack(blocks).astype(np.float32)
    meta = pd.DataFrame(rows, columns=META_COLS)

    # A NaN reaching the loss silently poisons every gradient after it.
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return WindowSet(X=X, y=meta["y"].to_numpy(np.float32), meta=meta, features=features)


def group_split(windows: WindowSet, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """Split on duels, not on windows.

    Windows from one duel overlap in time and share both trajectories, so a window-level
    split puts near-copies of the test set into training and reports a score that will
    not survive contact with a new capture.
    """

    total = cfg.train + cfg.val + cfg.test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    folds = (
        windows.meta.groupby(FOLD_COLS)["y"]
        .max()
        .reset_index()
        .sort_values(FOLD_COLS)
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(cfg.seed)
    assignment: dict[tuple, str] = {}

    # Stratified on whether the duel contained a cheater, so a small capture cannot
    # hand one fold every cheating episode by chance.
    for _, part in folds.groupby("y"):
        keys = part[FOLD_COLS].itertuples(index=False, name=None)
        keys = np.array(list(keys), dtype=object)
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(round(n * cfg.train))
        n_val = int(round(n * cfg.val))
        n_train = min(n_train, max(n - 2, 0)) if n >= 3 else n_train

        for i, key in enumerate(keys):
            name = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            assignment[tuple(key)] = name

    keys = list(windows.meta[FOLD_COLS].itertuples(index=False, name=None))
    where = np.array([assignment[k] for k in keys])

    return {name: np.flatnonzero(where == name) for name in ("train", "val", "test")}


def describe_split(windows: WindowSet, splits: dict[str, np.ndarray]) -> str:
    lines = [f"{'split':<6} {'windows':>8} {'cheat':>7} {'rate':>6} {'duels':>6}"]
    for name, idx in splits.items():
        meta = windows.meta.iloc[idx]
        pos = int(meta["y"].sum())
        duels = len(meta.groupby(FOLD_COLS)) if len(meta) else 0
        rate = pos / len(idx) if len(idx) else 0.0
        lines.append(f"{name:<6} {len(idx):>8} {pos:>7} {rate:>6.3f} {duels:>6}")
    return "\n".join(lines)
