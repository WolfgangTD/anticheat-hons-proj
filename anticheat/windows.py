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
from .data import (
    CHEAT_COLUMNS,
    FOLD_COLS,
    GROUP_COLS,
    HEADS,
    MASK_COLS,
    TARGET_COLS,
    PAD_FEATURE,
    Capture,
    feature_columns,
    label_table,
)

# Episode-level targets travel in the metadata, beside the window-level ones in
# WindowSet.y: scoring a whole player pools every window, so it needs the episode's
# answer, not the per-window one.
EPISODE_TARGET_COLS = [f"e{c}" for c in TARGET_COLS]
EPISODE_MASK_COLS = [f"e{c}" for c in MASK_COLS]

META_COLS = (GROUP_COLS + ["anchor", "anchor_kind", "label_type",
                           "label_wallhack", "label_aim_snap", "label_trigger_bot"]
             + EPISODE_TARGET_COLS + EPISODE_MASK_COLS)

CHEATING = HEADS.index("cheating")
CHEAT_HEADS = [(head, HEADS.index(head)) for head in CHEAT_COLUMNS]


@dataclass
class WindowSet:
    X: np.ndarray            # (N, T, F) float32
    y: np.ndarray            # (N, H) float32, one column per entry in HEADS
    mask: np.ndarray         # (N, H) float32, 0 where that head has no valid target
    meta: pd.DataFrame       # one row per window, carries the fold keys
    features: list[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray, with_x: bool = True) -> "WindowSet":
        """Rows `idx`. Without X, only the labels and metadata are copied: scoring reads
        windows straight from the full array, so a fold never needs its own copy."""

        return WindowSet(
            X=self.X[idx] if with_x else self.X[:0],
            y=self.y[idx],
            mask=self.mask[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            features=self.features,
        )


def _anchor_indices(shot: np.ndarray, damage: np.ndarray, los: np.ndarray, cfg: WindowConfig) -> list[tuple[int, str]]:
    """Positions within one agent-episode that are worth centring a window on."""

    found: list[tuple[int, str]] = []

    for kind in cfg.anchors:
        if kind == "shot":
            hits = np.flatnonzero(shot == 1)
        elif kind == "damage":
            hits = np.flatnonzero(damage > 0)
        elif kind == "los_acquire":
            hits = np.flatnonzero((los == 1) & (np.r_[0, los[:-1]] == 0))
        elif kind == "uniform":
            hits = np.arange(0, len(shot), cfg.stride)
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


def _firing_index(capture: Capture) -> dict[tuple, np.ndarray]:
    """(session, episode_id, agent_id, cheat) -> sorted decisions where that cheat fired."""

    if capture.activations is None:
        return {}
    grouped = capture.activations.groupby(GROUP_COLS + ["cheat"], sort=False)["decision"]
    return {key: np.sort(ticks.to_numpy()) for key, ticks in grouped}


def _fired_between(ticks: np.ndarray | None, first: int, last: int) -> bool:
    if ticks is None:
        return False
    i = np.searchsorted(ticks, first)
    return bool(i < len(ticks) and ticks[i] <= last)


def _window_labels(
    episode_target: np.ndarray,
    episode_mask: np.ndarray,
    firings: dict[str, np.ndarray | None],
    first: int,
    last: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window targets for the cheat heads; the cheating head keeps the episode's.

    Running the cheat and firing inside the window is a positive. Running it without
    firing inside the window is masked: that stretch is the agent's own play, and calling
    it honest would be as wrong as calling it cheating. Not running the cheat at all is
    a clean negative.
    """

    target, mask = episode_target.copy(), episode_mask.copy()
    for head, h in CHEAT_HEADS:
        if episode_target[h] == 0:
            continue
        hit = _fired_between(firings[head], first, last)
        target[h] = 1.0 if hit else 0.0
        mask[h] = 1.0 if hit else 0.0
    return target, mask


def build_windows(capture: Capture, data_cfg: DataConfig, cfg: WindowConfig, x_path: str | None = None) -> WindowSet:
    """Cut every anchored window out of the capture.

    Two passes: the first finds every anchor, so the window array is allocated once at
    its final size; the second fills it. With `x_path` the array is a .npy file on disk
    rather than memory, which is how a multi-hour capture stays inside the RAM budget.
    """

    features = feature_columns(data_cfg)
    pad_col = features.index(PAD_FEATURE)
    labels = label_table(capture.episodes, data_cfg)
    fired = _firing_index(capture)
    per_window = capture.activations is not None

    steps = capture.steps
    values = steps[features].to_numpy(dtype=np.float32)
    decisions = steps["decision"].to_numpy()
    shot = steps["shot"].to_numpy()
    damage = steps["damage_dealt"].to_numpy()
    los = steps["has_los"].to_numpy()
    sessions = steps["session"].to_numpy()
    episode_ids = steps["episode_id"].to_numpy()
    agent_ids = steps["agent_id"].to_numpy()

    # load_capture sorts by agent-episode then decision, so each one is a contiguous run
    # of rows and can be sliced rather than grouped.
    group_id = steps.groupby(GROUP_COLS, sort=False).ngroup().to_numpy()
    starts = np.flatnonzero(np.r_[True, group_id[1:] != group_id[:-1]])
    ends = np.r_[starts[1:], len(steps)]

    label_row = {key: i for i, key in enumerate(labels.index)}
    episode_targets = labels[TARGET_COLS].to_numpy(dtype=np.float32)
    episode_masks = labels[MASK_COLS].to_numpy(dtype=np.float32)

    plan = []
    for start, end in zip(starts, ends):
        key = (sessions[start], episode_ids[start], agent_ids[start])
        row = label_row.get(key)
        if row is None:
            continue
        anchors = _anchor_indices(shot[start:end], damage[start:end], los[start:end], cfg)
        if anchors:
            plan.append((key, row, start, end, anchors))

    n_windows = sum(len(p[4]) for p in plan)
    if n_windows == 0:
        raise RuntimeError("no windows produced; check the anchor settings")

    shape = (n_windows, cfg.length, len(features))
    if x_path:
        X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=shape)
    else:
        X = np.empty(shape, dtype=np.float32)

    y = np.empty((n_windows, len(HEADS)), dtype=np.float32)
    mask = np.empty_like(y)
    window_row = np.empty(n_windows, dtype=np.int64)
    anchor = np.empty(n_windows, dtype=np.int64)
    kinds = np.empty(n_windows, dtype=object)
    offsets = np.arange(-cfg.pre + 1, cfg.post + 1)

    k = 0
    for key, row, start, end, anchors in plan:
        n = end - start
        group_values = values[start:end]
        group_decisions = decisions[start:end]
        firings = {head: fired.get((*key, head)) for head, _ in CHEAT_HEADS}

        for idx, kind in anchors:
            take = idx + offsets
            out_of_range = (take < 0) | (take >= n)

            # Near an episode boundary the window is completed by repeating the edge
            # frame, and the padding is flagged so the model can discount it rather
            # than reading a fabricated freeze as behaviour.
            window = group_values[np.clip(take, 0, n - 1)]
            window[out_of_range, pad_col] = 1.0

            # A NaN reaching the loss silently poisons every gradient after it.
            np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            X[k] = window

            if per_window:
                # Padding repeats the edge frame, so only the real ticks can hold a firing.
                first = group_decisions[max(idx - cfg.pre + 1, 0)]
                last = group_decisions[min(idx + cfg.post, n - 1)]
                y[k], mask[k] = _window_labels(episode_targets[row], episode_masks[row], firings, first, last)
            else:
                y[k], mask[k] = episode_targets[row], episode_masks[row]

            window_row[k] = row
            anchor[k] = idx
            kinds[k] = kind
            k += 1

    if x_path:
        X.flush()

    episodes = labels.reset_index()
    meta = episodes.iloc[window_row][
        GROUP_COLS + ["label_type", "label_wallhack", "label_aim_snap", "label_trigger_bot"]
    ].reset_index(drop=True)
    meta["anchor"] = anchor
    meta["anchor_kind"] = kinds
    meta[EPISODE_TARGET_COLS] = episode_targets[window_row]
    meta[EPISODE_MASK_COLS] = episode_masks[window_row]

    return WindowSet(X=X, y=y, mask=mask, meta=meta[META_COLS], features=features)


def group_split(windows: WindowSet, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """Split on duels, not on windows.

    Windows from one duel overlap in time and share both trajectories, so a window-level
    split puts near-copies of the test set into training and reports a score that will
    not survive contact with a new capture.
    """

    total = cfg.train + cfg.val + cfg.test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    frame = windows.meta[FOLD_COLS].copy()
    frame["y"] = windows.y[:, CHEATING]

    folds = (
        frame.groupby(FOLD_COLS)["y"]
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
    """Window counts per fold, with the positives each head will actually train on."""

    cheats = list(CHEAT_COLUMNS)
    lines = [f"{'split':<6} {'windows':>8} {'cheat':>7} {'rate':>6} {'duels':>6}   "
             + "  ".join(f"{c:>10}" for c in cheats)]

    for name, idx in splits.items():
        meta = windows.meta.iloc[idx]
        pos = int(windows.y[idx, CHEATING].sum())
        duels = len(meta.groupby(FOLD_COLS)) if len(meta) else 0
        rate = pos / len(idx) if len(idx) else 0.0

        # Masked windows are excluded, so this is the count the head's loss sees.
        per_head = [
            int((windows.y[idx, HEADS.index(c)] * windows.mask[idx, HEADS.index(c)]).sum())
            for c in cheats
        ]
        lines.append(f"{name:<6} {len(idx):>8} {pos:>7} {rate:>6.3f} {duels:>6}   "
                     + "  ".join(f"{n:>10}" for n in per_head))
    return "\n".join(lines)
