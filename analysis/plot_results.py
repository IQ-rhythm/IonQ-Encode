import os, json
import matplotlib.pyplot as plt

ENCODERS = ["angle", "amplitude_exact", "amplitude_approx", "dru", "qks", "kfm"]
TITLE_SUFFIX = "T2 PCA16" 

def load_logs(enc):
    path = f"results/logs/{enc}_results.json"
    if not os.path.exists(path):
        print(f"⚠️ Skip {enc}: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)

def plot_metric(metric_key, title):
    any_plotted = False
    for enc in ENCODERS:
        log = load_logs(enc)
        if not log or metric_key not in log:
            continue
        plt.plot(log[metric_key], label=enc)
        any_plotted = True
    if not any_plotted:
        print(f"❗ No data to plot for '{metric_key}'")
        return
    plt.xlabel("epoch")
    plt.ylabel(metric_key)
    plt.legend()
    plt.title(f"{title} — {TITLE_SUFFIX}")
    plt.tight_layout()
    plt.show()

# 1) Validation Accuracy
plot_metric("val_acc", "Validation Accuracy")

# 2) Validation F1
plot_metric("val_f1", "Validation F1")
