#!/usr/bin/env python3
"""
Batch launcher for Phase 1 Fitting Dynamics experiments.
Dispatches 35 runs across 4 GPUs using a simple job queue.
"""
import subprocess
import time
from pathlib import Path

NUM_GPUS = 4
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "FittingDynamics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
SEEDS = [42, 123, 456]

# ─── Job queue ──────────────────────────────────────────────────────────
jobs = []

# Block A: SIREN scratch (15 runs)
for img in IMAGES:
    for seed in SEEDS:
        out = RESULTS_DIR / f"SIREN_{img}_seed{seed}"
        cmd = (
            f"python experiments/Phase1_FittingDynamics/run_siren.py "
            f"--image Data/Set5/HR/{img}.png --seed {seed} "
            f"--save_dir {out}"
        )
        jobs.append(("A", f"SIREN_{img}_s{seed}", cmd))

# Block B: LIIF scratch SR x4 (5 runs)
for img in IMAGES:
    out = RESULTS_DIR / f"LIIF_{img}_seed42"
    cmd = (
        f"python experiments/Phase1_FittingDynamics/run.py "
        f"--model liif --image Data/Set5/HR/{img}.png --sr 4 "
        f"--seed 42 --save_dir {out}"
    )
    jobs.append(("B", f"LIIF_{img}_s42", cmd))

# Block C: LTE scratch SR x4 (5 runs)
for img in IMAGES:
    out = RESULTS_DIR / f"LTE_{img}_seed42"
    cmd = (
        f"python experiments/Phase1_FittingDynamics/run.py "
        f"--model lte --image Data/Set5/HR/{img}.png --sr 4 "
        f"--seed 42 --save_dir {out}"
    )
    jobs.append(("C", f"LTE_{img}_s42", cmd))

# Block D: PretrainedLIIF fine-tune SR x4 (5 runs)
for img in IMAGES:
    out = RESULTS_DIR / f"PretrainedLIIF_{img}_seed42"
    cmd = (
        f"python experiments/Phase1_FittingDynamics/run_finetune.py "
        f"--model_type liif --image Data/Set5/HR/{img}.png "
        f"--scale 4 --lr_size 48 --steps 5000 --seed 42 --save_dir {out}"
    )
    jobs.append(("D1", f"PtLIIF_{img}_s42", cmd))

# Block D: PretrainedLIIF_EQ fine-tune SR x4 (5 runs)
for img in IMAGES:
    out = RESULTS_DIR / f"PretrainedLIIF_EQ_{img}_seed42"
    cmd = (
        f"python experiments/Phase1_FittingDynamics/run_finetune.py "
        f"--model_type liif_eq --image Data/Set5/HR/{img}.png "
        f"--scale 4 --lr_size 48 --steps 5000 --seed 42 --save_dir {out}"
    )
    jobs.append(("D2", f"PtEQ_{img}_s42", cmd))

N = len(jobs)

# ─── Helpers ────────────────────────────────────────────────────────────

def gpu_usage():
    """Return sorted list of (gpu_index, used_memory_mib)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,nounits,noheader"], text=True
        )
        rows = [(int(idx), int(mem)) for idx, mem in
                (line.split(", ") for line in out.strip().split("\n"))]
        rows.sort(key=lambda x: x[1])
        return rows
    except Exception:
        return [(i, 0) for i in range(NUM_GPUS)]


def pick_gpu():
    return gpu_usage()[0][0]


def reap_finished(active, completed_log):
    finished = []
    for pid, (block, name, gpu, start, proc) in list(active.items()):
        if proc.poll() is not None:
            rc = proc.returncode
            elapsed = time.time() - start
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{status}] {name} GPU{gpu} "
                  f"{int(elapsed//60)}m{int(elapsed%60)}s")
            finished.append(pid)
            with open(completed_log, "a") as f:
                f.write(f"{name}\n")
    for pid in finished:
        del active[pid]


# ─── Queue runner ───────────────────────────────────────────────────────
completed_log = RESULTS_DIR / ".completed_jobs.txt"
completed = set()
if completed_log.exists():
    completed = set(completed_log.read_text().strip().split("\n"))

print(f"Total: {N}, Completed: {len(completed)}, Remaining: {N - len(completed)}")

active = {}
for i, (block, name, cmd) in enumerate(jobs):
    tag = f"{name}"
    if tag in completed:
        print(f"[SKIP {i+1}/{N}] {tag}")
        continue

    gpu = pick_gpu()
    full_cmd = f"cd {BASE_DIR} && CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    logfile = RESULTS_DIR / f"{name}.log"

    print(f"[LAUNCH {i+1}/{N}] {tag} → GPU {gpu}  log={logfile.name}")
    proc = subprocess.Popen(
        full_cmd, shell=True, stdout=open(logfile, "w"), stderr=subprocess.STDOUT
    )
    active[proc.pid] = (block, name, gpu, time.time(), proc)
    (RESULTS_DIR / f".{name}.pid").write_text(str(proc.pid))

    # Wait for a slot if all GPUs busy
    while len(active) >= NUM_GPUS:
        reap_finished(active, completed_log)
        if len(active) >= NUM_GPUS:
            time.sleep(15)

# Wait for stragglers
print(f"\nWaiting for {len(active)} remaining...")
while active:
    reap_finished(active, completed_log)
    if active:
        time.sleep(30)

print(f"\n{'='*60}")
print(f"All {N} experiments complete!")
print(f"Logs: {RESULTS_DIR}/*.log")
print(f"{'='*60}")
