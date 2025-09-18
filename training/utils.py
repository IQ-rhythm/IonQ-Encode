import torch
from sklearn.metrics import accuracy_score, f1_score
import json
import os

def bce_loss_with_logits():
    return torch.nn.BCEWithLogitsLoss()

def compute_metrics(y_true, y_pred_logits):
    """
    y_true: numpy array (0/1)
    y_pred_logits: numpy array of logits (real values)
    """
    y_pred = (y_pred_logits >= 0).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"accuracy": acc, "f1": f1}

def save_log(log_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(log_dict, f, indent=4)
