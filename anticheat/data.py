"""Loading and feature engineering for the shooter's telemetry capture.

The capture is deliberately split in two: `steps.csv` holds only what an authoritative
server can observe, `episodes.csv` holds the ground-truth cheat severities. This module
is the only place the two meet, and the labels never reach the feature matrix.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig

GROUP_COLS = ["session", "episode_id", "agent_id"]
# Both agents of a duel share one trajectory, so they must land in the same fold.
FOLD_COLS = ["session", "episode_id"]

# The model's outputs, in order. "cheating" is the IDS alarm the evaluation leads with;
# the other three say which cheat. Multi-label, not multiclass: one agent can run
# several cheats at once, so each head is its own binary target.
HEADS = ("cheating", "wallhack", "aimsnap", "triggerbot")

# Per-cheat head -> (severity label, ground-truth activation counter) in episodes.csv.
CHEAT_COLUMNS = {
    "wallhack": ("label_wallhack", "gt_wallhack_reveals"),
    "aimsnap": ("label_aim_snap", "gt_aimsnap_adjustments"),
    "triggerbot": ("label_trigger_bot", "gt_triggerbot_shots"),
}

TARGET_COLS = [f"y_{head}" for head in HEADS]
MASK_COLS = [f"m_{head}" for head in HEADS]

# ShotResult in Weapon.cs: NotReady=0, Miss=1, Hit=2. The writer emits -1 for "no shot".
SHOT_MISS = 1
SHOT_HIT = 2

# Arena-absolute geometry. Informative (line of sight follows the map) but it also lets
# the model memorise spawns, so the "relative" feature set drops it.
ABSOLUTE_FEATURES = ["pos_x", "pos_z", "enemy_pos_x", "enemy_pos_z", "yaw_sin", "yaw_cos"]

BASE_FEATURES = [
    # Self kinematics
    "vel_x",
    "vel_z",
    "speed",
    "turn_deg",
    # Duel geometry
    "dist",
    "d_dist",
    "rel_fwd",
    "rel_right",
    # Aim
    "aim_err_sin",
    "aim_err_cos",
    "aim_error_abs",
    "aim_error_delta_deg",
    "aim_speed",
    # Visibility. has_los is rare (~11% of steps), which is what makes tracking an
    # enemy through a wall such a clean wallhack tell.
    "has_los",
    "los_state_age_s",
    "aim_speed_no_los",
    "aim_error_abs_no_los",
    # Target acquisition. Dwell before a shot is the trigger-bot tell: a human needs
    # reaction time between the target entering the cone and the trigger, a bot does not.
    "target_in_cone",
    "cone_dwell_s",
    "cone_dwell_at_shot",
    "los_age_at_shot",
    # Firing
    "weapon_ready",
    "shot",
    "shot_hit",
    "shot_miss",
    "damage_dealt",
    # Both players' state
    "health",
    "enemy_health",
    "enemy_speed",
    "enemy_yaw_rel_sin",
    "enemy_yaw_rel_cos",
]

# Marks frames that were edge-replicated because the anchor sat near an episode boundary.
PAD_FEATURE = "is_pad"


def feature_columns(cfg: DataConfig) -> list[str]:
    cols = list(BASE_FEATURES)
    if cfg.feature_set == "full":
        cols = ABSOLUTE_FEATURES + cols
    elif cfg.feature_set != "relative":
        raise ValueError(f"unknown feature_set {cfg.feature_set!r}, expected full|relative")
    return cols + [PAD_FEATURE]


ACTIVATION_COLS = GROUP_COLS + ["decision", "cheat"]

# Identity columns keep integer and string types; every measurement is read as float32,
# which halves the frame against pandas' float64 default and is what the model consumes.
KEY_DTYPES = {"session": str, "agent_id": str, "episode_id": np.int64, "arena_id": np.int64, "decision": np.int64}


def _read_steps(path: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: KEY_DTYPES.get(c, np.float32) for c in columns})


@dataclass
class Capture:
    steps: pd.DataFrame
    episodes: pd.DataFrame
    activations: pd.DataFrame | None  # one row per cheat firing; None with window_labels off
    dropped_groups: int  # step groups with no matching episode row
    filtered_episodes: int  # cheating episodes dropped because the cheat never fired


def load_capture(cfg: DataConfig) -> Capture:
    """Read every session under the telemetry root and join steps to their labels."""

    root = cfg.telemetry_root
    step_files = sorted(glob.glob(os.path.join(root, "session_*", "steps.csv")))
    episode_files = sorted(glob.glob(os.path.join(root, "session_*", "episodes.csv")))

    if not step_files:
        raise FileNotFoundError(f"no session_*/steps.csv under {root}")

    steps = pd.concat([_read_steps(f) for f in step_files], ignore_index=True)
    episodes = pd.concat([pd.read_csv(f) for f in episode_files], ignore_index=True)

    steps["session"] = steps["session"].astype(str)
    episodes["session"] = episodes["session"].astype(str)

    activations = None
    if cfg.window_labels:
        session_dirs = [os.path.dirname(f) for f in step_files]
        missing = [d for d in session_dirs if not os.path.isfile(os.path.join(d, "activations.csv"))]
        if missing:
            raise FileNotFoundError(
                f"window labels need activations.csv, which these sessions predate: {missing}. "
                "Pass --episode-labels to label every window with its episode's draw instead."
            )
        activations = pd.concat(
            [pd.read_csv(os.path.join(d, "activations.csv"), usecols=ACTIVATION_COLS) for d in session_dirs],
            ignore_index=True,
        )
        activations["session"] = activations["session"].astype(str)

    # A run cut off mid-episode leaves steps with no FinishEpisode row. Those windows
    # have no label, so they are dropped rather than guessed at.
    keys = episodes.set_index(GROUP_COLS).index
    before = len(steps.groupby(GROUP_COLS, sort=False))
    steps = steps[steps.set_index(GROUP_COLS).index.isin(keys)].reset_index(drop=True)
    dropped = before - len(steps.groupby(GROUP_COLS, sort=False))

    filtered = 0
    if cfg.require_fired:
        fired = pd.Series(False, index=episodes.index)
        for label, gt in CHEAT_COLUMNS.values():
            fired |= (episodes[label] > 0) & (episodes[gt] > 0)
        drop = (episodes["label_cheating"] == 1) & ~fired
        filtered = int(drop.sum())
        if filtered:
            episodes = episodes[~drop].reset_index(drop=True)
            keys = episodes.set_index(GROUP_COLS).index
            steps = steps[steps.set_index(GROUP_COLS).index.isin(keys)].reset_index(drop=True)

    steps = steps.sort_values(GROUP_COLS + ["decision"]).reset_index(drop=True)
    steps = engineer(steps)

    return Capture(steps=steps, episodes=episodes, activations=activations,
                   dropped_groups=dropped, filtered_episodes=filtered)


def engineer(steps: pd.DataFrame) -> pd.DataFrame:
    """Derive the behavioural features the raw log only implies.

    Nothing here consults a label. Angles become sin/cos so that the wrap at +/-180
    is not read as a huge jump, and the cheat tells are made explicit as products
    rather than left for the sequence model to discover from a pilot-sized capture.

    Works in place: on a multi-hour capture a defensive copy of the frame costs gigabytes.
    """

    df = steps
    g = df.groupby(GROUP_COLS, sort=False)

    yaw = np.deg2rad(df["yaw_deg"].to_numpy())
    df["yaw_sin"] = np.sin(yaw)
    df["yaw_cos"] = np.cos(yaw)

    aim = np.deg2rad(df["aim_error_deg"].to_numpy())
    df["aim_err_sin"] = np.sin(aim)
    df["aim_err_cos"] = np.cos(aim)
    df["aim_error_abs"] = df["aim_error_deg"].abs()

    rel_yaw = np.deg2rad(df["enemy_yaw_rel_deg"].to_numpy())
    df["enemy_yaw_rel_sin"] = np.sin(rel_yaw)
    df["enemy_yaw_rel_cos"] = np.cos(rel_yaw)

    # Enemy offset in the agent's own frame: how far in front, how far to the side.
    dx = df["enemy_pos_x"] - df["pos_x"]
    dz = df["enemy_pos_z"] - df["pos_z"]
    df["rel_fwd"] = dz * np.cos(yaw) + dx * np.sin(yaw)
    df["rel_right"] = dx * np.cos(yaw) - dz * np.sin(yaw)

    # Closing speed. First step of an episode has no predecessor, hence the fill.
    df["d_dist"] = g["dist"].diff().fillna(0.0)

    df["aim_speed"] = df["aim_error_delta_deg"].abs()

    # The wallhack signature: aim that tracks a target the agent cannot legitimately see.
    no_los = 1.0 - df["has_los"]
    df["aim_speed_no_los"] = df["aim_speed"] * no_los
    df["aim_error_abs_no_los"] = df["aim_error_abs"] * no_los

    # The trigger-bot signature: the trigger pulled the instant the target crossed the cone.
    df["cone_dwell_at_shot"] = df["cone_dwell_s"] * df["shot"]
    df["los_age_at_shot"] = df["los_state_age_s"] * df["shot"]

    # shot_result is an enum, so the ordinal is meaningless as a number.
    df["shot_hit"] = (df["shot_result"] == SHOT_HIT).astype(np.float32)
    df["shot_miss"] = (df["shot_result"] == SHOT_MISS).astype(np.float32)

    df[PAD_FEATURE] = np.float32(0.0)
    return df


def label_table(episodes: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Per-agent-episode labels, indexed for window lookup.

    Adds a target and a mask column per head. A mask of 0 means the head has no valid
    target for that episode and is left out of its loss and its metrics.
    """

    cols = GROUP_COLS + [
        "label_cheating",
        "label_type",
        "label_wallhack",
        "label_aim_snap",
        "label_trigger_bot",
        "gt_wallhack_reveals",
        "gt_aimsnap_adjustments",
        "gt_triggerbot_shots",
        "outcome",
        "decisions",
    ]
    table = episodes[cols].copy()

    table["y_cheating"] = (table["label_cheating"] == 1).astype(np.float32)
    table["m_cheating"] = np.float32(1.0)

    for head, (label, gt) in CHEAT_COLUMNS.items():
        drawn = table[label] > 0
        table[f"y_{head}"] = drawn.astype(np.float32)

        # The episode-level filter in load_capture only drops cheaters where nothing
        # fired. In a multi-cheat episode one cheat can fire while another never does;
        # the episode stays, and only the silent cheat's head is told to ignore it,
        # rather than being taught either "cheating" or "honest" from no evidence.
        unfired = drawn & (table[gt] == 0) if cfg.require_fired else pd.Series(False, index=table.index)
        table[f"m_{head}"] = (~unfired).astype(np.float32)

    return table.set_index(GROUP_COLS)


def summarise(capture: Capture) -> str:
    eps, steps = capture.episodes, capture.steps
    lines = [
        f"sessions          {eps['session'].nunique()}",
        f"episodes          {len(eps)} agent-episodes over {len(eps.groupby(FOLD_COLS))} duels",
        f"steps             {len(steps):,}",
        f"dropped groups    {capture.dropped_groups} (steps with no episode row)",
        f"filtered episodes {capture.filtered_episodes} (cheat drawn but never fired)",
        "cheat heads       " + ("labelled per window from activations.csv" if capture.activations is not None
                                else "labelled per episode (--episode-labels)"),
        "",
        "label_cheating    " + str(eps["label_cheating"].value_counts().to_dict()),
        "label_type        " + str(eps["label_type"].value_counts().to_dict()),
        "",
        f"shots             {int((steps['shot'] == 1).sum()):,}",
        f"has_los rate      {steps['has_los'].mean():.3f}",
    ]
    return "\n".join(lines)
