import json, glob, os
rows = []
for path in glob.glob("results/logs/*_results.json"):
    with open(path) as f:
        log = json.load(f)
    enc = os.path.basename(path).replace("_results.json","")
    best_val_acc = max(log["val_acc"]) if log["val_acc"] else float("nan")
    best_val_f1  = max(log["val_f1"])  if log["val_f1"]  else float("nan")
    rows.append((enc, best_val_acc, best_val_f1))
rows.sort(key=lambda x: x[1], reverse=True)
print(f"{'encoder':20s}  best_val_acc  best_val_f1")
for enc, acc, f1 in rows:
    print(f"{enc:20s}  {acc:12.4f}  {f1:11.4f}")
