import json, matplotlib.pyplot as plt
encoders = ["angle","amplitude_exact","amplitude_approx","dru","qks","kfm"]
for enc in encoders:
    with open(f"results/logs/{enc}_results.json") as f:
        log = json.load(f)
    plt.plot(log["val_acc"], label=enc)
plt.xlabel("epoch"); plt.ylabel("val_acc"); plt.legend(); plt.title("T2 PCA32")
plt.show()
