"""Score a model trained with exp.py on val, internal test and the three external sets, then compare it with F.
  .venv/bin/python eval_exp.py full_F3_aux --K 1 [--comb mlp] [--aux]
Scores go to results_final/scores/<name>.npz (same layout evaluate_new.py writes, so report_new.evaluate reads them);
metrics to results_exp/<name>_metrics.json. The threshold is the max-MCC one on val, applied unchanged everywhere."""
import argparse
import json
from pathlib import Path

import torch

import exp
from evaluate_new import SC, save, sets
from report_new import SETS, evaluate

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--K", type=int, default=1)
p.add_argument("--comb", default="mlp")
p.add_argument("--aux", action="store_true")
a = p.parse_args()

SC.mkdir(parents=True, exist_ok=True)
data = sets()
model = exp.Ctx(a.K, a.comb, a.aux).to(exp.DEV)
model.load_state_dict(torch.load(exp.OUT / f"{a.name}.pt", map_location=exp.DEV))
save(a.name, {k: exp.predict(model, ws, exp.pointers(ws, a.K)) for k, ws in data.items()}, data)

new, _ = evaluate(SC / f"{a.name}.npz")
old, _ = evaluate(SC / "F_full.npz")
(exp.OUT / f"{a.name}_metrics.json").write_text(json.dumps(new, indent=1))

print(f"{'set':8s} {'metric':14s} {'F (old)':>9s} {'new':>9s} {'diff':>8s}")
print(f"{'val':8s} {'auroc':14s} {old['val_auroc']:9.3f} {new['val_auroc']:9.3f} {new['val_auroc'] - old['val_auroc']:+8.3f}")
for k in SETS:
    for view in ("all", "sleep_only"):
        for m in ("auroc", "mcc", "sensitivity", "burden_spearman"):
            o, n = old[k][view][m], new[k][view][m]
            print(f"{k:8s} {view[:5] + ' ' + m[:8]:14s} {o:9.3f} {n:9.3f} {n - o:+8.3f}")
