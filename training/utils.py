import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import os

def bce_loss_with_logits():
    return torch.nn.BCEWithLogitsLoss()

def compute_metrics(y_true, y_pred_logits):
    """
    Compute accuracy and F1 score from logits.
    
    Args:
        y_true: list, numpy array, or torch.Tensor of true labels (0/1)
        y_pred_logits: list, numpy array, or torch.Tensor of logits (real values)
    
    Returns:
        dict: {"accuracy": float, "f1": float}
    """
    # Convert inputs to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.detach().cpu().numpy()
    elif isinstance(y_pred_logits, list):
        y_pred_logits = np.array(y_pred_logits)

    # Binary prediction from logits (threshold at 0)
    y_pred = (y_pred_logits >= 0).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"accuracy": acc, "f1": f1}

def save_log(log_dict, save_path):
    """
    Save training log as JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(log_dict, f, indent=4)
