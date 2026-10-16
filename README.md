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
python -m anticheat train --out runs/hybrid
```

```bash
python -m anticheat eval runs/hybrid --split test
```

`inspect` prints the capture summary, the feature list and the fold sizes without
training anything — run it first after any new capture.

## Layout

| File | What it holds |
| --- | --- |
| `config.py` | Every knob, as nested dataclasses. Written to `config.json` beside each checkpoint. |
| `data.py` | Session loading, the steps/episodes join, and feature engineering. |
| `windows.py` | Event-anchored window extraction and group-aware splits. |
| `model.py` | `HybridDetector`: the LSTM backbone with transformer encoder blocks on it, four heads. |
| `train.py` | Standardiser, training loop, early stopping, checkpointing. |
| `metrics.py` | ROC/PR, FPR-at-recall, base-rate correction — numpy, no sklearn dependency. |
| `evaluate.py` | The IDS-style report, window level and agent-episode level. |
| `cli.py` | `inspect` / `train` / `eval`. |

## The model

One network: transformer technology applied on an LSTM. The two are not alternatives
and are not compared against each other.

```
window (80 ticks x 38 features)
  -> biLSTM, 2 layers                 reads the window tick by tick
  -> transformer encoder x 2          self-attention + feed-forward over the LSTM's outputs
  -> attention pooling                one learned weight per tick
  -> shared trunk -> 4 binary heads   cheating | wallhack | aimsnap | triggerbot
```

The LSTM carries the short-range dynamics (a snap is a few ticks of angular velocity, a
trigger pull is the tick a target crosses the cone), and because it is recurrent its
outputs already encode order, so no positional encoding is added. The encoder lets every
tick attend to every other, which relates events far apart in the window: line of sight
opening and the shot that follows it, or aim that kept tracking a target the agent could
not see. The pooling weights come back from `forward(x, return_attention=True)`, one per
tick, to be checked against the ticks in `activations.csv` where a cheat actually fired.

**The heads are multi-label, not multiclass.** One agent can run several cheats at
once, so each head is an independent sigmoid. `cheating` is the IDS alarm the evaluation
leads with; the other three say which cheat. `label_type = "multi"` is a convenience
string in the capture, not a class.

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
checks the `gt_*` counters and filters those out; `--keep-unfired` disables it. In a
multi-cheat episode where one cheat fired and another never did, the episode stays and
only the silent cheat's head is masked out of the loss and the metrics. Real datasets
cannot do this — it is an advantage of generating the data.

**The cheat heads are labelled per window, from `activations.csv`.** A window is a
positive for a cheat only if that cheat fired inside it. Windows from an agent running
the cheat where it did not fire are masked, neither positive nor negative; agents not
running it are clean negatives. Episode labels alone fail badly for trigger-bot: it
fires only on an on-target opportunity, so a trigger-bot episode is mostly the agent's
own shots, and in the 2026-10-16 capture only 22% of trigger-bot-labelled shot windows
held a bot firing (10% at severity 0.2-0.4). The bot-fired shots themselves are
distinctive — 53% land on the tick the target enters the aim cone, against 5% of
ordinary on-target shots — so the signal is there once the labels point at it. The
`cheating` head stays per episode, and player-level scoring pools every window against
the episode's label, since a deployed detector cannot know which windows hold a firing.
`--episode-labels` restores episode labels for sessions captured before the sidecar.

**Model selection is on average precision, not AUC.** With rare positives AUC stays
comfortable while precision collapses, and precision is what decides whether an honest
player gets banned. It is the mean over the four heads, so the checkpoint kept cannot
buy a better alarm by giving up the trigger-bot head.

**The deployment threshold is chosen on validation.** `evaluate` picks each head's
threshold to hit 90% recall on the validation split and then applies it to test, rather
than reading the best operating point off the split being reported.

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
- **Every head separately**, plus the alarm's recall by what the agent was running, so a
  headline number cannot hide one cheat being missed entirely.
- **Recall by severity** for aimsnap and triggerbot, so the result reads "detects
  trigger-bot down to severity X". Wallhack is binary by design and has no severity axis.

## Pilot result

From an earlier single-output LSTM, before the model above, on the three sessions captured 2026-09-06 — 277 agent-episodes, 95k steps, 4,561
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

- Capture at scale with the config recorded in `meta.json` (every arena on multi-cheat,
  trigger-bot severity 0.2-1).
- Check the pooling attention against `activations.csv`: within a positive window, does
  the model look at the ticks where the cheat fired?
- Choose the held-out split (unseen `agent_id`, or by session) before reading any
  validation number.

## Reference

Loo, Lužkov & Burelli, *AntiCheatPT: A Transformer-Based Approach to Cheat Detection in
Competitive Computer Games*, IEEE Conference on Games 2025. arXiv:2508.06348.
