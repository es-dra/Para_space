"""Weight-space classification of SIREN zoos with and without exact program augmentation (stage 07).

  python scripts/train_wsl.py --train zoo/mnist_s32_train5k --test zoo/mnist_s32_test1k --n-train 1000 \
      --model deepsets2 --aug programs --seed 0 --epochs 60 --out results/wsl/deepsets2_programs_1000_s0.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from wsinr import siren, wsl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--model", choices=["deepsets2", "neuralgraph", "aligned_mlp"], default="neuralgraph")
    ap.add_argument("--aug", choices=["none", "programs", "permute"], default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr_effs, ytr = wsl.load_zoo(a.train, a.n_train)
    te_effs, yte = wsl.load_zoo(a.test, a.n_test)
    if a.model == "aligned_mlp":
        ref = tr_effs[0]
        tr_effs = [siren.to_effective(p) for p in wsl.align_batch(tr_effs, ref)]
        te_effs = [siren.to_effective(p) for p in wsl.align_batch(te_effs, ref)]
    Etr, Ete = wsl.stack(tr_effs), wsl.stack(te_effs)
    ytr_t, yte_t = torch.tensor(ytr), torch.tensor(yte)
    n_params = sum(v[0].numel() for v in Etr.values())
    model = {"deepsets2": lambda: wsl.DeepSets2(d=a.width), "neuralgraph": lambda: wsl.NeuralGraph(d=a.width)}.get(a.model, lambda: wsl.AlignedMLP(n_params))()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(ytr) / a.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.CrossEntropyLoss()

    def prep(E):
        return wsl.canonicalize_signs(E) if a.model in ("deepsets2", "neuralgraph") else E

    def evaluate():
        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(yte), 256):
                Eb = {k: v[i:i + 256] for k, v in Ete.items()}
                correct += (model(prep(Eb)).argmax(1) == yte_t[i:i + 256]).sum().item()
        model.train()
        return correct / len(yte)

    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        tot, nb = 0.0, 0
        for i in range(0, len(ytr), a.batch):
            idx = perm[i:i + a.batch]
            Eb = {k: v[idx] for k, v in Etr.items()}
            if a.aug == "programs":
                Eb = wsl.augment_programs(Eb, rng=rng)
            elif a.aug == "permute":
                Eb = wsl.augment_permute(Eb, rng=rng)
            if a.model == "aligned_mlp" and a.aug == "programs":
                pass  # programs act on layer 1 and the readout only; alignment of hidden units is unchanged
            loss = lossf(model(prep(Eb)), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            acc = evaluate()
            hist.append({"epoch": ep + 1, "loss": tot / nb, "test_acc": acc})
            print(f"ep {ep + 1:3d} loss {tot / nb:.3f} test acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": vars(a), "n_train": int(len(ytr)), "n_test": int(len(yte)), "final_test_acc": hist[-1]["test_acc"],
               "history": hist, "seconds": time.time() - t0}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
