# anticheat

Behavioural cheat detection over server-side telemetry from the 1v1 shooter at
`../test`. The shooter generates the data; this trains and evaluates the detector.

No frame buffers, no client trust, no signatures. The input is what an authoritative
server can already see: both players' transforms and velocities, line of sight traced
against world geometry, and the shots that arrive.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
python -m anticheat inspect
```

```bash
python -m anticheat train --out runs/lstm
```

```bash
python -m anticheat eval runs/lstm --split test
```

```bash
python -m anticheat compare --out runs/compare
```

`inspect` prints the capture summary, the feature list and the fold sizes without
training anything — run it first after any new capture.

## Layout

| File | What it holds |
| --- | --- |
| `config.py` | Every knob, as nested dataclasses. Written to `config.json` beside each checkpoint. |
| `data.py` | Session loading, the steps/episodes join, and feature engineering. |
| `windows.py` | Event-anchored window extraction and group-aware splits. |
| `model.py` | `LSTMDetector` (the committed architecture) and `TransformerDetector` (baseline). |
| `train.py` | Standardiser, training loop, early stopping, checkpointing. |
| `metrics.py` | ROC/PR, FPR-at-recall, base-rate correction — numpy, no sklearn dependency. |
| `evaluate.py` | The IDS-style report, window level and agent-episode level. |
| `cli.py` | `inspect` / `train` / `eval` / `compare`. |

## Design decisions

**Windows are anchored on events, not slid over the episode.** Cheats only express
themselves in engagements; a window centred on an agent walking through an empty arena
teaches the model nothing. Default anchor is every shot, 64 steps before and 16 after
(1.6 s at 50 Hz). `--anchors shot los_acquire damage uniform` to widen it. This is
borrowed from AntiCheatPT, which anchors 256 ticks on each kill.

**Folds are cut on duels, never on windows.** Both agents in an episode share one
trajectory and consecutive windows overlap in time, so any finer split puts near-copies
of the test set into training. `group_split` stratifies duels on whether they contained
a cheater and assigns every window of a duel to the same fold.

**Imbalance is handled by weighting the loss, not by augmentation.** AntiCheatPT
duplicates minority windows with Gaussian noise added to the coordinates. That inflates
the dataset with near-identical rows and makes the split boundary load-bearing in a way
that is easy to get wrong. `pos_weight = n_neg / n_pos` costs nothing and leaves the
data honest.

**Episodes where a cheat was drawn but never fired are dropped by default.** The
severity draw labels an episode, but a wallhack that never revealed anything produced
no behaviour to detect, so that episode is a mislabelled positive. `require_fired`
checks the `gt_*` counters and filters those out; `--keep-unfired` disables it. Real
datasets cannot do this — it is an advantage of generating the data.

**Model selection is on average precision, not AUC.** With rare positives AUC stays
comfortable while precision collapses, and precision is what decides whether an honest
player gets banned.

**The deployment threshold is chosen on validation.** `evaluate` picks the threshold
that hits 90% recall on the validation split and then applies it to test, rather than
reading the best operating point off the split being reported.

## What the evaluation reports

Accuracy on a balanced test set is the number that flatters a detector most and means
least. The report leads with:

- **FPR at fixed recall** and **recall at fixed FPR** — the operator's actual dial.
- **Precision at realistic prevalence.** The capture is ~48% cheating because the
  simulator draws cheats that often. Real populations are 1–5%. The table restates
  precision for that world; it is usually a much worse number, and it is the honest one.
- **Agent-episode aggregation.** Windows are pooled per player by mean, max, top-k mean
  and alarm rate. AntiCheatPT names single-event judgement as its own main limitation:
  a 3% per-window false-positive rate compounds to roughly a coin flip across a
  20-engagement match. Aggregation is what makes the rate survivable.
- **Recall by cheat type**, so a headline number cannot hide one cheat being missed
  entirely.

## Pilot result

From the three sessions captured 2026-09-06 — 277 agent-episodes, 95k steps, 4,561
shot-anchored windows, 38 features:

```
                     window AUC   window AP   episode AUC (mean)
LSTM (best epoch 5)      0.9386      0.9564               0.9502
```

At the 90%-recall threshold: recall 0.943, FPR 0.338, and precision falls from 0.74 at
50% prevalence to **0.027 at 1% prevalence**. Per cheat: aimsnap 1.00, wallhack 0.97,
triggerbot 0.73.

Read this as a pipeline smoke test, not a result. 139 duels from three sessions of a
partly-trained policy is far too small to conclude anything, and the trigger-bot gap
is the first thing worth chasing with a bigger capture.

## Next

- Capture at scale, with the policy trained to convergence.
- `compare` with matched epoch budgets, to settle LSTM vs transformer on this data —
  the comparison AntiCheatPT never ran.
- Per-decision cheat-fired labels as a third sidecar, keyed on
  `(session, episode_id, agent_id, decision)`, to move from episode-level labels to
  instance-level ones. Labels stay out of the feature stream.
- Continual learning: add god mode, speed hacking and rapid trigger, then measure
  whether the detector adapts without forgetting the first three.

## Reference

Loo, Lužkov & Burelli, *AntiCheatPT: A Transformer-Based Approach to Cheat Detection in
Competitive Computer Games*, IEEE Conference on Games 2025. arXiv:2508.06348.
