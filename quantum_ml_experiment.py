#!/usr/bin/env python3
"""
Simple working demonstration of the quantum ML training pipeline.

This simplified version focuses on getting a working demonstration with 
practical computational requirements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure directories
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(dataset_name: str):
    """Load preprocessed dataset."""
    path = os.path.join(DATA_DIR, f"{dataset_name}.npz")
    data = np.load(path)
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']


def create_simple_ae_circuit(n_qubits: int, n_layers: int = 2):
    """Create a simple angle encoding circuit with practical qubit count."""
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="autograd")
    def circuit(features, weights):
        # Angle encoding - map features to first few qubits
        for i in range(min(len(features), n_qubits)):
            qml.RY(features[i], wires=i)
        
        # Variational layers
        for layer in range(n_layers):
            # Parameterized rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            
            # Entangling gates
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])  # Circular entanglement
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


def preprocess_features(x, target_dim=4):
    """Reduce feature dimensionality using simple selection."""
    if x.shape[1] > target_dim:
        # Select features with highest variance
        variances = np.var(x, axis=0)
        top_indices = np.argsort(variances)[-target_dim:]
        return x[:, top_indices]
    return x


def run_simple_experiment(dataset_name: str = "fashion_mnist_pca32_T2", 
                         n_qubits: int = 4, epochs: int = 10):
    """Run a simple quantum ML experiment."""
    print(f"Running Simple Quantum ML Experiment")
    print(f"Dataset: {dataset_name}")
    print(f"Qubits: {n_qubits}, Epochs: {epochs}")
    print("=" * 50)
    
    # Load and preprocess data
    print("1. Loading data...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)
    
    # Reduce dimensionality for practical computation
    x_train = preprocess_features(x_train, target_dim=n_qubits)
    x_test = preprocess_features(x_test, target_dim=n_qubits)
    
    # Create train/validation split
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"Train: {x_train_split.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    
    # Create quantum circuit
    print("2. Setting up quantum circuit...")
    circuit = create_simple_ae_circuit(n_qubits, n_layers=2)
    
    # Initialize weights
    n_layers = 2
    weights = np.random.normal(0, 0.1, (n_layers, n_qubits, 2))
    
    # Setup optimizer
    optimizer = AdamOptimizer(stepsize=0.01)
    
    # Training
    print("3. Training...")
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    batch_size = 32
    
    def cost_function(w, x_batch, y_batch):
        predictions = []
        for x_sample in x_batch:
            pred_raw = circuit(x_sample, w)
            pred = 1.0 / (1.0 + np.exp(-pred_raw))  # Sigmoid
            predictions.append(pred)
        
        predictions = np.array(predictions)
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)
        loss = -np.mean(y_batch * np.log(predictions) + 
                       (1 - y_batch) * np.log(1 - predictions))
        return loss
    
    def evaluate(x, y, w):
        predictions = []
        for x_sample in x:
            pred_raw = circuit(x_sample, w)
            pred = 1 if pred_raw > 0 else 0
            predictions.append(pred)
        return accuracy_score(y, predictions)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        n_batches = len(x_train_split) // batch_size
        epoch_loss = 0
        
        for i in range(min(n_batches, 10)):  # Limit batches for demo
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x_train_split))
            
            x_batch = x_train_split[start_idx:end_idx]
            y_batch = y_train_split[start_idx:end_idx]
            
            weights, loss = optimizer.step_and_cost(
                lambda w: cost_function(w, x_batch, y_batch), weights
            )
            epoch_loss += loss
        
        # Evaluation
        if epoch % 2 == 0:  # Evaluate every 2 epochs for speed
            train_acc = evaluate(x_train_split[:100], y_train_split[:100], weights)
            val_acc = evaluate(x_val, y_val, weights)
            
            history['train_loss'].append(epoch_loss / min(n_batches, 10))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch:2d}: Loss={epoch_loss/min(n_batches, 10):.4f}, "
                  f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("4. Final evaluation...")
    test_predictions = []
    for x_sample in x_test:
        pred_raw = circuit(x_sample, weights)
        pred = 1 if pred_raw > 0 else 0
        test_predictions.append(pred)
    
    test_acc = accuracy_score(y_test, test_predictions)
    
    # Results
    print("5. Results:")
    print(f"   Final Test Accuracy: {test_acc:.4f}")
    print(f"   Training Time: {training_time:.1f} seconds")
    
    # Save visualization
    exp_dir = os.path.join(RESULTS_DIR, f"simple_ae_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], 'b-o', label='Train', markersize=4)
    plt.plot(history['val_acc'], 'r-o', label='Validation', markersize=4)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title(f'Training Curves - Test Acc: {test_acc:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], 'g-o', markersize=4)
    plt.xlabel('Evaluation Step') 
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(exp_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save summary report
    report = {
        'experiment': {
            'dataset': dataset_name,
            'n_qubits': n_qubits,
            'n_layers': 2,
            'epochs': epochs,
            'feature_dim': x_train.shape[1]
        },
        'results': {
            'test_accuracy': float(test_acc),
            'final_train_acc': float(history['train_acc'][-1]) if history['train_acc'] else 0,
            'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0,
            'training_time': float(training_time)
        }
    }
    
    import json
    report_path = os.path.join(exp_dir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Results saved to: {exp_dir}")
    
    return test_acc, training_time


def run_comparison_experiments():
    """Run experiments comparing different configurations."""
    print("Running Comparison Experiments")
    print("=" * 60)
    
    configs = [
        ("fashion_mnist_pca32_T2", 4, 10),
        ("fashion_mnist_pca32_T2", 6, 10),
        ("fashion_mnist_pca32_T4", 4, 15),
    ]
    
    results = []
    
    for dataset, n_qubits, epochs in configs:
        print(f"\nConfiguration: {dataset} with {n_qubits} qubits, {epochs} epochs")
        try:
            acc, time_taken = run_simple_experiment(dataset, n_qubits, epochs)
            results.append({
                'dataset': dataset,
                'n_qubits': n_qubits,
                'epochs': epochs,
                'accuracy': acc,
                'time': time_taken,
                'status': 'success'
            })
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                'dataset': dataset,
                'n_qubits': n_qubits, 
                'epochs': epochs,
                'accuracy': 0,
                'time': 0,
                'status': 'failed'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"{status} {result['dataset'][:20]:20} | {result['n_qubits']}Q | "
              f"Acc: {result['accuracy']:.3f} | Time: {result['time']:.1f}s")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Quantum ML Demo")
    parser.add_argument('--dataset', default='fashion_mnist_pca32_T2')
    parser.add_argument('--qubits', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--comparison', action='store_true', 
                       help='Run comparison experiments')
    
    args = parser.parse_args()
    
    if args.comparison:
        run_comparison_experiments()
    else:
        run_simple_experiment(args.dataset, args.qubits, args.epochs)