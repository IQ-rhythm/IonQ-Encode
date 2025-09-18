import torch
from training.utils import compute_metrics

def evaluate(model, weights, dataloader, device="cpu"):
    model.set_params(weights)  # 최신 가중치 반영
    model.circuit.to(device)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    metrics = compute_metrics(all_labels, all_logits)
    return metrics
