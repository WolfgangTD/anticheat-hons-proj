#!/bin/bash
# Train the detector on the UTS eResearch HPC (PBS Pro). Submit from the repo root:
#
#   qsub -v DATA=$HOME/telemetry,RUN=hybrid-5h lab/train.sh
#
# DATA is the folder in your home directory holding the session_* folders; RUN names
# the output folder, runs/RUN. Extra `anticheat train` flags go in EXTRA and win over
# the defaults below, for example a second seed:
#
#   qsub -v DATA=$HOME/telemetry,RUN=hybrid-5h-seed2,EXTRA="--seed 2" lab/train.sh
#
# The capture is copied to the node's local /scratch and the windows are cached there,
# as the HPC asks for large files. Results are small and are written straight to
# runs/RUN in home, so nothing needs copying back, and a job killed at its walltime
# loses at most one epoch: submit the same command again and it resumes.
#
# Memory: the first pass over the CSVs peaks at about 3 GB per hour of capture, plus a
# few GB. 32GB covers roughly 9 hours of capture; raise mem for a longer one.
#PBS -N anticheat
#PBS -q gpuq
#PBS -l ngpus=1
#PBS -l ncpus=4
#PBS -l mem=32GB
#PBS -l walltime=04:00:00
# For an email when the job starts and ends, remove one # from each line and fill in:
##PBS -m abe
##PBS -M your.email@student.uts.edu.au
set -euo pipefail

: "${DATA:?pass DATA, e.g. qsub -v DATA=\$HOME/telemetry,RUN=name lab/train.sh}"
: "${RUN:?pass RUN, e.g. qsub -v DATA=\$HOME/telemetry,RUN=name lab/train.sh}"

cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# The same path on every submission of this run, so a resumed job's settings match the
# checkpoint's. Scratch is local to each node, so a resume on another node re-copies.
WORK="/scratch/${USER}/anticheat/${RUN}"
mkdir -p "$WORK/telemetry" "runs/$RUN"
trap 'rm -rf "$WORK"' EXIT
cp -rp "$DATA"/session_* "$WORK/telemetry/"

# Batch 512 keeps a GPU busy where the default 128 suits a CPU.
python -m anticheat train \
    --telemetry "$WORK/telemetry" \
    --out "runs/$RUN" \
    --cache-dir "$WORK/cache" \
    --resume \
    --epochs 40 \
    --batch-size 512 \
    ${EXTRA:-} 2>&1 | tee -a "runs/$RUN/train.log"
