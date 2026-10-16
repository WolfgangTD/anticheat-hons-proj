# Training on the lab machines

The lab has NVIDIA GPUs and around 64 GB of RAM per machine; data and code go up over
SSH. Replace `USER@HOST` with your lab login. Commands marked **PowerShell** run on your
own PC; the rest run on the lab machine.

## 1. Upload the capture

Pack the session folders that make up the dataset (every one under the folder becomes
training data, so leave out anything recorded with different settings):

**PowerShell**
```powershell
tar -czf capture.tgz -C E:\pp-test\test\Telemetry session_20261015_233109_099_19412 session_20261016_010952_195_19412
```

```powershell
scp capture.tgz USER@HOST:~/
```

```powershell
ssh USER@HOST "mkdir -p ~/telemetry && tar -xzf ~/capture.tgz -C ~/telemetry"
```

CSV compresses about five to one, so this is far quicker than copying the folders as
they are.

## 2. Upload the code

`git archive` packs exactly what is committed, without runs, caches or weights:

**PowerShell**
```powershell
git -C E:\pp-test\anticheat archive -o anticheat.tgz HEAD
```

```powershell
scp anticheat.tgz USER@HOST:~/
```

```powershell
ssh USER@HOST "mkdir -p ~/anticheat && tar -xzf ~/anticheat.tgz -C ~/anticheat"
```

## 3. Set up once

```bash
cd ~/anticheat && bash lab/setup.sh
```

It ends by printing the GPU it found.

## 4. Train

**If you log straight into a machine**, run inside `tmux` so the job survives the SSH
connection dropping (`Ctrl+B` then `D` to detach, `tmux attach -t train` to come back):

```bash
tmux new -s train
```

```bash
cd ~/anticheat && bash lab/train.sh ~/telemetry hybrid-5h
```

**If the lab uses a SLURM queue**, submit from the repo root instead; check the
`#SBATCH` lines at the top of `lab/train.sh` against the lab's limits first:

```bash
cd ~/anticheat && sbatch lab/train.sh ~/telemetry hybrid-5h
```

Either way:

- Progress goes to `runs/hybrid-5h/train.log` (`tail -f` it). `history.json` is
  rewritten after every epoch with train and validation loss per head, ready to plot.
- If the job is killed, run the same command again: it resumes after the last finished
  epoch.
- The first run on a capture builds the windows into `cache/`. Every later run on the
  same data (another seed, more epochs) starts training within seconds.
- Training ends with the evaluation report, written to `runs/hybrid-5h/metrics_test.json`.

Extra flags go after the run name and override the defaults, for example a second seed
for error bars:

```bash
bash lab/train.sh ~/telemetry hybrid-5h-seed2 --seed 2
```

## 5. Bring the results back

**PowerShell**
```powershell
scp -r USER@HOST:~/anticheat/runs/hybrid-5h E:\pp-test\anticheat\runs\
```
