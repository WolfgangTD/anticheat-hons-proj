"""Command line entry point.

    python -m anticheat inspect
    python -m anticheat train --out runs/hybrid
    python -m anticheat eval runs/hybrid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .evaluate import evaluate
from .train import prepare, train
from .windows import describe_split


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--telemetry", help="root holding session_*/ capture directories")
    parser.add_argument("--feature-set", choices=["full", "relative"])
    parser.add_argument("--keep-unfired", action="store_true",
                        help="keep cheating episodes whose cheat never actually fired")
    parser.add_argument("--episode-labels", action="store_true",
                        help="label the cheat heads per episode, for sessions without activations.csv")
    parser.add_argument("--cache-dir", help="keep built windows here and reuse them on later runs")

    parser.add_argument("--pre", type=int, help="steps before the anchor")
    parser.add_argument("--post", type=int, help="steps after the anchor")
    parser.add_argument("--anchors", nargs="+", choices=["shot", "los_acquire", "damage", "uniform"])
    parser.add_argument("--min-gap", type=int)

    parser.add_argument("--hidden", type=int, help="LSTM hidden size per direction")
    parser.add_argument("--layers", type=int, help="LSTM layers")
    parser.add_argument("--encoder-layers", type=int, help="transformer encoder layers on the LSTM")
    parser.add_argument("--attn-heads", type=int, help="attention heads per encoder layer")
    parser.add_argument("--ff-dim", type=int, help="encoder feed-forward width")
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)


def _config_from(args: argparse.Namespace) -> Config:
    cfg = Config()

    if args.telemetry:
        cfg.data.telemetry_root = args.telemetry
    if args.feature_set:
        cfg.data.feature_set = args.feature_set
    if args.keep_unfired:
        cfg.data.require_fired = False
    if args.episode_labels:
        cfg.data.window_labels = False
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir

    if args.pre is not None:
        cfg.window.pre = args.pre
    if args.post is not None:
        cfg.window.post = args.post
    if args.anchors:
        cfg.window.anchors = tuple(args.anchors)
    if args.min_gap is not None:
        cfg.window.min_gap = args.min_gap

    if args.hidden is not None:
        cfg.model.hidden = args.hidden
    if args.layers is not None:
        cfg.model.layers = args.layers
    if args.encoder_layers is not None:
        cfg.model.encoder_layers = args.encoder_layers
    if args.attn_heads is not None:
        cfg.model.attn_heads = args.attn_heads
    if args.ff_dim is not None:
        cfg.model.ff_dim = args.ff_dim
    if args.dropout is not None:
        cfg.model.dropout = args.dropout

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.patience is not None:
        cfg.train.patience = args.patience
    if args.device:
        cfg.train.device = args.device
    if args.seed is not None:
        cfg.train.seed = args.seed
        cfg.split.seed = args.seed

    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anticheat", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="summarise the capture, the windows and the splits")
    _add_common(p_inspect)

    p_train = sub.add_parser("train", help="train a detector")
    p_train.add_argument("--out", default="runs/latest", help="output directory")
    p_train.add_argument("--resume", action="store_true",
                         help="continue from <out>/last.pt if a previous run there was cut off")
    _add_common(p_train)

    p_eval = sub.add_parser("eval", help="score a trained run")
    p_eval.add_argument("run_dir")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])

    args = parser.parse_args(argv)

    if args.command == "inspect":
        cfg = _config_from(args)
        summary, windows, splits = prepare(cfg)
        print(summary)
        print()
        print(f"windows           {len(windows)} x {windows.X.shape[1]} steps x {windows.X.shape[2]} features")
        print(f"features          {', '.join(windows.features)}")
        print()
        print(describe_split(windows, splits))
        return 0

    if args.command == "train":
        cfg = _config_from(args)
        train(cfg, Path(args.out), resume=args.resume)
        print()
        evaluate(Path(args.out), split="test")
        return 0

    if args.command == "eval":
        evaluate(Path(args.run_dir), split=args.split)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
