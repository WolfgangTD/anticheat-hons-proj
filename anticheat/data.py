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


@dataclass
class Capture:
    steps: pd.DataFrame
    episodes: pd.DataFrame
    dropped_groups: int  # step groups with no matching episode row
    filtered_episodes: int  # cheating episodes dropped because the cheat never fired


def load_capture(cfg: DataConfig) -> Capture:
    """Read every session under the telemetry root and join steps to their labels."""

    root = cfg.telemetry_root
    step_files = sorted(glob.glob(os.path.join(root, "session_*", "steps.csv")))
    episode_files = sorted(glob.glob(os.path.join(root, "session_*", "episodes.csv")))

    if not step_files:
        raise FileNotFoundError(f"no session_*/steps.csv under {root}")

    steps = pd.concat([pd.read_csv(f) for f in step_files], ignore_index=True)
    episodes = pd.concat([pd.read_csv(f) for f in episode_files], ignore_index=True)

    steps["session"] = steps["session"].astype(str)
    episodes["session"] = episodes["session"].astype(str)

    # A run cut off mid-episode leaves steps with no FinishEpisode row. Those windows
    # have no label, so they are dropped rather than guessed at.
    keys = episodes.set_index(GROUP_COLS).index
    before = len(steps.groupby(GROUP_COLS, sort=False))
    steps = steps[steps.set_index(GROUP_COLS).index.isin(keys)].reset_index(drop=True)
    dropped = before - len(steps.groupby(GROUP_COLS, sort=False))

    filtered = 0
    if cfg.require_fired:
        fired = (
            ((episodes["label_wallhack"] > 0) & (episodes["gt_wallhack_reveals"] > 0))
            | ((episodes["label_aim_snap"] > 0) & (episodes["gt_aimsnap_adjustments"] > 0))
            | ((episodes["label_trigger_bot"] > 0) & (episodes["gt_triggerbot_shots"] > 0))
        )
        drop = (episodes["label_cheating"] == 1) & ~fired
        filtered = int(drop.sum())
        if filtered:
            episodes = episodes[~drop].reset_index(drop=True)
            keys = episodes.set_index(GROUP_COLS).index
            steps = steps[steps.set_index(GROUP_COLS).index.isin(keys)].reset_index(drop=True)

    steps = steps.sort_values(GROUP_COLS + ["decision"]).reset_index(drop=True)
    steps = engineer(steps)

    return Capture(steps=steps, episodes=episodes, dropped_groups=dropped, filtered_episodes=filtered)


def engineer(steps: pd.DataFrame) -> pd.DataFrame:
    """Derive the behavioural features the raw log only implies.

    Nothing here consults a label. Angles become sin/cos so that the wrap at +/-180
    is not read as a huge jump, and the cheat tells are made explicit as products
    rather than left for the sequence model to discover from a pilot-sized capture.
    """

    df = steps.copy()
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

    df[PAD_FEATURE] = 0.0
    return df


def label_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per-agent-episode labels, indexed for window lookup."""

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
    return episodes[cols].set_index(GROUP_COLS)


def summarise(capture: Capture) -> str:
    eps, steps = capture.episodes, capture.steps
    lines = [
        f"sessions          {eps['session'].nunique()}",
        f"episodes          {len(eps)} agent-episodes over {len(eps.groupby(FOLD_COLS))} duels",
        f"steps             {len(steps):,}",
        f"dropped groups    {capture.dropped_groups} (steps with no episode row)",
        f"filtered episodes {capture.filtered_episodes} (cheat drawn but never fired)",
        "",
        "label_cheating    " + str(eps["label_cheating"].value_counts().to_dict()),
        "label_type        " + str(eps["label_type"].value_counts().to_dict()),
        "",
        f"shots             {int((steps['shot'] == 1).sum()):,}",
        f"has_los rate      {steps['has_los'].mean():.3f}",
    ]
    return "\n".join(lines)
