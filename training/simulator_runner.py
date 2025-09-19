import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from quantum_encodings.angle_encoding import AngleEncodingClassifier
from quantum_encodings.amplitude_encoding import AmplitudeEncodingClassifier
from quantum_encodings.hybrid_encoding import HybridEncodingClassifier
from training.utils import bce_loss_with_logits, compute_metrics, save_log


def load_npz_dataset(path):
    data = np.load(path)
    x_train = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    x_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    return x_train, y_train, x_test, y_test


def build_model(args, n_features):
    encoder_name = args.encoder

    if encoder_name == "angle":
        return AngleEncodingClassifier(n_features=n_features, n_layers=args.n_layers)
    elif encoder_name == "amplitude_exact":
        return AmplitudeEncodingClassifier(n_features=n_features, n_layers=args.n_layers, method="exact")
    elif encoder_name == "amplitude_approx":
        return AmplitudeEncodingClassifier(n_features=n_features, n_layers=args.n_layers, method="approximate")
    elif encoder_name == "hybrid":
        n_angle_features = min(8, n_features)
        n_amplitude_log = int(np.ceil(np.log2(n_features - n_angle_features)))
        return HybridEncodingClassifier(
            n_angle_features=n_angle_features,
            n_amplitude_features_log=n_amplitude_log,
            n_layers=args.n_layers,
            entanglement_strategy="linear"
        )
    # elif encoder_name == "qks":
        # QKS: Quantum Kitchen Sink
        # n_qubits = n_features
        # return EnsembleQKS(n_qubits=n_qubits, n_layers=args.n_layers, n_features=n_features)
    # elif encoder_name == "dru":
    #     # DRU: Data Re-Uploading
    #     n_qubits = n_features
    #     return DRUClassifier(n_qubits=n_qubits, n_layers=args.n_layers)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")



def train(args):
    # === Load dataset ===
    x_train, y_train, x_test, y_test = load_npz_dataset(args.dataset)
    # For quick testing, you can uncomment the following lines to use a smaller subset
    x_train, y_train = x_train[:100], y_train[:100]
    x_test, y_test   = x_test[:20], y_test[:20]

    # Only binary classification supported for now
    if len(torch.unique(y_train)) > 2:
        raise ValueError("Only binary classification is supported in this pipeline.")

    # Wrap datasets
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    # === Build model ===
    n_features = x_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, n_features)
    weights = model.get_params().clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([weights], lr=args.lr)
    loss_fn = bce_loss_with_logits()

    logs = {"train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": []}

    # === Training loop ===
    for epoch in range(args.epochs):
        model.set_params(weights)
        model.weights.requires_grad_(True)

        epoch_loss, all_logits, all_labels = 0, [], []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch).squeeze()
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_metrics = compute_metrics(all_labels, all_logits)

        # Validation (test set as proxy)
        with torch.no_grad():
            val_logits, val_labels = [], []
            for x_batch, y_batch in test_loader:
                val_logits_batch = model(x_batch).squeeze()
                val_logits.extend(val_logits_batch.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

            val_metrics = compute_metrics(val_labels, val_logits)
            val_loss = loss_fn(torch.tensor(val_logits), torch.tensor(val_labels, dtype=torch.float32)).item()

        # Logging
        logs["train_loss"].append(epoch_loss / len(train_loader))
        logs["train_acc"].append(train_metrics["accuracy"])
        logs["train_f1"].append(train_metrics["f1"])
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_metrics["accuracy"])
        logs["val_f1"].append(val_metrics["f1"])

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

    # Save results
    save_log(logs, f"./results/logs/{args.encoder}_results.json")

    print("Training complete. Results saved to /results/logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to .npz dataset (e.g., data/processed/fashion_mnist_pca16_T2.npz)")
    parser.add_argument("--encoder", type=str, default="angle",
                        choices=["angle", "amplitude_exact", "amplitude_approx", "hybrid", "qks", "dru"],)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", type=int, default=2)
    args = parser.parse_args()

    train(args)
