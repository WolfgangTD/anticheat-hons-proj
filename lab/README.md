# Training on the lab machines

The UTS eResearch HPC runs PBS Pro, with GPUs in the `gpuq` queue. Data and code go up
over SSH to your home directory. Replace `USER@HOST` with your HPC login. Commands
marked **PowerShell** run on your own PC; the rest run on the HPC.

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

On the login node (installing is fine there; training is not):

```bash
cd ~/anticheat && bash lab/setup.sh
```

The login node has no GPU, so the check at the end saying none is visible is expected.

## 4. Train

Submit from the repo root:

```bash
cd ~/anticheat && qsub -v DATA=$HOME/telemetry,RUN=hybrid-5h lab/train.sh
```

- `qstat` shows the job (`Q` queued, `R` running); `qdel <job id>` cancels it.
- Progress goes to `runs/hybrid-5h/train.log` (`tail -f` it). PBS also writes
  `anticheat.o<id>` and `anticheat.e<id>` where you ran `qsub`; the `.e` file should be
  empty.
- `history.json` is rewritten after every epoch with train and validation loss per
  head, ready to plot.
- If the job hits its walltime, submit the same command again: it resumes after the
  last finished epoch.
- Training ends with the evaluation report, in `runs/hybrid-5h/metrics_test.json`.
- The resource requests are at the top of `lab/train.sh`. Raise `mem` there if the
  capture is longer than about 9 hours.

Another seed, for error bars:

```bash
qsub -v DATA=$HOME/telemetry,RUN=hybrid-5h-seed2,EXTRA="--seed 2" lab/train.sh
```

## 5. Bring the results back

**PowerShell**
```powershell
scp -r USER@HOST:~/anticheat/runs/hybrid-5h E:\pp-test\anticheat\runs\
```
