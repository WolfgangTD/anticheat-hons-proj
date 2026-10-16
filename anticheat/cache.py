"""On-disk cache of built windows.

Reading the CSVs and cutting windows out of a multi-hour capture takes minutes and most
of the memory, and every run after the first (another seed, more epochs, evaluation)
needs exactly the same result. The cache is keyed on the capture files and on every
setting that changes the windows, so a new capture or a different window length can
never pick up a stale one.

Built windows are opened memory-mapped, read-only: training reads them through the
operating system's page cache instead of holding a private copy per process.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_capture, summarise
from .windows import WindowSet, build_windows

# Bump when windowing or labelling changes in a way the settings do not capture.
CACHE_VERSION = 1


def cache_key(cfg: Config) -> str:
    files = []
    for session in sorted(glob.glob(os.path.join(cfg.data.telemetry_root, "session_*"))):
        for name in ("steps.csv", "episodes.csv", "activations.csv"):
            path = os.path.join(session, name)
            if os.path.isfile(path):
                stat = os.stat(path)
                files.append([os.path.basename(session), name, stat.st_size, int(stat.st_mtime)])

    # Where the capture lives and where the cache lives do not change the windows.
    data = {k: v for k, v in asdict(cfg.data).items() if k not in ("telemetry_root", "cache_dir")}
    material = {"version": CACHE_VERSION, "files": files, "data": data, "window": asdict(cfg.window)}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def load_or_build(cfg: Config) -> tuple[str, WindowSet]:
    """The capture summary and its windows, from the cache when one is configured."""

    if not cfg.data.cache_dir:
        capture = load_capture(cfg.data)
        return summarise(capture), build_windows(capture, cfg.data, cfg.window)

    path = Path(cfg.data.cache_dir) / cache_key(cfg)
    if (path / "complete").is_file():
        info = json.loads((path / "info.json").read_text(encoding="utf-8"))
        windows = WindowSet(
            X=np.load(path / "X.npy", mmap_mode="r"),
            y=np.load(path / "y.npy"),
            mask=np.load(path / "mask.npy"),
            meta=pd.read_pickle(path / "meta.pkl"),
            features=info["features"],
        )
        return f"{info['summary']}\nwindows from cache {path}", windows

    path.mkdir(parents=True, exist_ok=True)
    capture = load_capture(cfg.data)
    summary = summarise(capture)
    windows = build_windows(capture, cfg.data, cfg.window, x_path=str(path / "X.npy"))
    del capture

    np.save(path / "y.npy", windows.y)
    np.save(path / "mask.npy", windows.mask)
    windows.meta.to_pickle(path / "meta.pkl")
    (path / "info.json").write_text(json.dumps({"features": windows.features, "summary": summary}), encoding="utf-8")

    # Written last, so a build that died partway is rebuilt rather than trusted.
    (path / "complete").write_text("", encoding="utf-8")

    windows.X = np.load(path / "X.npy", mmap_mode="r")
    return f"{summary}\nwindows cached to {path}", windows
