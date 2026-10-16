"""Configuration for the behavioural cheat detector.

Everything the pipeline needs to be reproduced lives in one nested dataclass, which
is written to `config.json` beside every checkpoint. A run is therefore described by
its output directory alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# anticheat/anticheat/config.py -> anticheat/ -> pp-test/
_REPO = Path(__file__).resolve().parents[1]
_WORKSPACE = _REPO.parent

DEFAULT_TELEMETRY = _WORKSPACE / "test" / "Telemetry"


@dataclass
class DataConfig:
    """Where the capture is and which columns become model input."""

    telemetry_root: str = str(DEFAULT_TELEMETRY)

    # "full" keeps arena-absolute positions; "relative" drops them so the detector
    # cannot memorise spawn points and has to work from the duel geometry alone.
    feature_set: str = "full"

    # Episodes are labelled by the severity draw, but a drawn cheat that never fired
    # produced no behaviour to detect. With this on, a cheating episode is kept only
    # if its matching gt_* counter is non-zero; honest episodes are never filtered.
    require_fired: bool = True

    # Label the per-cheat heads per window from activations.csv: a window is positive
    # only if that cheat fired inside it, and windows from an episode running the cheat
    # where it did not fire are masked, not called honest. A trigger-bot episode is
    # mostly the agent's own shots, so labelling every window with the episode's draw
    # teaches the head that ordinary shots are cheating. The cheating head stays
    # per-episode either way. Off is for sessions captured before the sidecar existed.
    window_labels: bool = True

    # Where built windows are kept between runs. Empty builds them in memory every time,
    # which is fine for a short capture and too slow and too large for a long one.
    cache_dir: str = ""


@dataclass
class WindowConfig:
    """Event-anchored context windows.

    AntiCheatPT anchors 256 ticks on each kill (224 before, 32 after) at 64 Hz. The
    shooter logs at 50 Hz and episodes run ~5 s, so the same 4 s window would swallow
    most of an episode. 64 before / 16 after is 1.6 s around each anchor.
    """

    pre: int = 64
    post: int = 16

    # shot | los_acquire | damage | uniform
    anchors: tuple[str, ...] = ("shot",)

    # Anchors closer together than this collapse to one window. Shots come every
    # ~10 steps at a 0.2 s cooldown, and overlapping windows are near-duplicates.
    min_gap: int = 10

    # Only used by the "uniform" anchor.
    stride: int = 25

    @property
    def length(self) -> int:
        return self.pre + self.post


@dataclass
class SplitConfig:
    """Group-aware splitting.

    Folds are cut on (session, episode_id), never on windows. Both agents of a duel
    share a trajectory, so splitting any finer leaks the test set into training.
    """

    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    seed: int = 42


@dataclass
class ModelConfig:
    """One model: an LSTM backbone with transformer encoder blocks on its outputs."""

    # LSTM backbone
    hidden: int = 128
    layers: int = 2
    bidirectional: bool = True

    # Transformer encoder over the LSTM's per-tick outputs. Its width is the LSTM's
    # output width (2 x hidden when bidirectional), so nothing is projected in between,
    # and that width must divide evenly across the attention heads.
    encoder_layers: int = 2
    attn_heads: int = 4
    ff_dim: int = 512

    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8
    # Positive class weight in the loss. "auto" uses n_neg / n_pos on the train split.
    pos_weight: str | float = "auto"
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42


@dataclass
class EvalConfig:
    threshold: float = 0.5
    # Operating points an IDS cares about, rather than a single accuracy number.
    recall_targets: tuple[float, ...] = (0.80, 0.90, 0.95)
    fpr_targets: tuple[float, ...] = (0.05, 0.01, 0.001)
    # Player-level precision depends on how rare cheaters actually are.
    prevalences: tuple[float, ...] = (0.5, 0.1, 0.05, 0.01)
    # Episode score = mean of the top-k window scores.
    topk: int = 3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return Config(
            data=DataConfig(**raw["data"]),
            window=WindowConfig(**{**raw["window"], "anchors": tuple(raw["window"]["anchors"])}),
            split=SplitConfig(**raw["split"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            eval=EvalConfig(
                **{
                    **raw["eval"],
                    "recall_targets": tuple(raw["eval"]["recall_targets"]),
                    "fpr_targets": tuple(raw["eval"]["fpr_targets"]),
                    "prevalences": tuple(raw["eval"]["prevalences"]),
                }
            ),
        )
