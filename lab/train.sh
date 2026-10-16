#!/usr/bin/env bash
# Train the detector on a lab GPU machine.
#
#   bash lab/train.sh DATA_DIR RUN_NAME [extra `anticheat train` flags...]
#   sbatch lab/train.sh DATA_DIR RUN_NAME [...]     # if the lab turns out to use SLURM
#
# DATA_DIR holds the session_* folders. Output goes to runs/RUN_NAME, including
# train.log. Built windows are cached in cache/ and reused by every later run on the
# same data, so only the first run pays for reading the CSVs.
#
# Re-running the same command after the job was killed (queue time limit, lost SSH
# session) resumes from the last finished epoch instead of starting over.
#
# The #SBATCH lines are plain comments to bash. Under SLURM they request one GPU; adjust
# them to whatever the lab's queue allows once that is known.
#SBATCH --job-name=anticheat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=anticheat-%j.log
set -euo pipefail

DATA=${1:?usage: lab/train.sh DATA_DIR RUN_NAME [flags...]}
RUN=${2:?usage: lab/train.sh DATA_DIR RUN_NAME [flags...]}
shift 2

# SLURM runs a copy of this script from its spool directory, so $0 does not point into
# the repo there; sbatch must be run from the repo root.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p "runs/$RUN"

# Batch 512 keeps a GPU busy where the default 128 suits a CPU. Later flags win, so
# anything passed after RUN_NAME overrides these.
python -m anticheat train \
    --telemetry "$DATA" \
    --out "runs/$RUN" \
    --cache-dir cache \
    --resume \
    --epochs 40 \
    --batch-size 512 \
    "$@" 2>&1 | tee -a "runs/$RUN/train.log"
