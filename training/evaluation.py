import torch
from training.utils import compute_metrics

def evaluate(model, weights, dataloader, device="cpu"):
    # 최신 가중치 반영
    model.set_params(weights)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            # torch.Tensor -> numpy 변환 (batch/스칼라 모두 대응)
            logits_np = logits.detach().cpu().numpy().flatten()
            labels_np = y_batch.detach().cpu().numpy().flatten()

            all_logits.extend(logits_np)
            all_labels.extend(labels_np)

    metrics = compute_metrics(all_labels, all_logits)
    return metrics
