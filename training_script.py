"""
Comprehensive Quantum Machine Learning Training Script for IonQ Encode Challenge.

This script implements the complete training pipeline following the research plan:
1. Data download and preprocessing (Fashion-MNIST)
2. Multiple quantum encoding methods (AE, DRU, AMP, Hybrid, Kernel, QKS)
3. Training with optimization and evaluation
4. Results reporting and visualization

Usage:
    python training_script.py --encoding ae --dataset fashion_mnist_pca32_T2 --epochs 20
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime

# Quantum ML imports
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

# Scikit-learn for classical methods and metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Local imports
from quantum_encodings import (
    AngleEncodingClassifier, AmplitudeEncodingClassifier, 
    HybridEncodingClassifier, QuantumKernelFeatureMap, 
    QuantumKitchenSinks
)

# Configure directories
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Dataset parameters
        self.dataset_name = "fashion_mnist_pca32_T2"
        self.test_split = 0.1
        self.val_split = 0.1
        
        # Training parameters
        self.epochs = 20
        self.batch_size = 32
        self.learning_rate = 5e-3
        self.lr_decay = 0.95
        self.optimizer = "adam"
        
        # Quantum parameters
        self.encoding_method = "ae"  # ae, dru, amp, hybrid, kernel, qks
        self.n_qubits = None  # Auto-determined from data
        self.n_layers = 3
        self.shots = 1000  # For hardware runs
        
        # Evaluation parameters
        self.metrics = ["accuracy", "f1"]
        self.verbose = True


class DataLoader:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed dataset from .npz file."""
        path = os.path.join(DATA_DIR, f"{dataset_name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {path}")
        
        data = np.load(path)
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']
    
    @staticmethod
    def create_train_val_split(x_train: np.ndarray, y_train: np.ndarray, 
                              val_split: float = 0.1) -> Tuple[np.ndarray, ...]:
        """Split training data into train and validation sets."""
        return train_test_split(x_train, y_train, test_size=val_split, 
                               random_state=42, stratify=y_train)


class ModelFactory:
    """Factory class for creating quantum models."""
    
    @staticmethod
    def create_model(encoding_method: str, n_features: int, n_classes: int, 
                    n_layers: int = 3, **kwargs) -> Any:
        """Create quantum model based on encoding method."""
        n_qubits = max(int(np.ceil(np.log2(n_features))), 2)
        
        if encoding_method == "ae":
            return AngleEncodingClassifier(
                n_qubits=n_qubits, 
                n_layers=n_layers,
                n_classes=n_classes
            )
        
        elif encoding_method == "dru":
            # Data re-uploading uses more qubits for expressivity
            n_qubits = min(n_features, 8)  # Limit for practical simulation
            from quantum_encodings.data_reuploading import build_dru_classifier
            circuit = build_dru_classifier(n_qubits, n_layers)
            return {"circuit": circuit, "n_qubits": n_qubits, "n_layers": n_layers}
        
        elif encoding_method == "amp":
            return AmplitudeEncodingClassifier(
                n_features=n_features,
                n_classes=n_classes,
                approximate=(n_features > 32)
            )
        
        elif encoding_method == "hybrid":
            return HybridEncodingClassifier(
                n_features=n_features,
                n_classes=n_classes,
                n_layers=n_layers
            )
        
        elif encoding_method == "kernel":
            return QuantumKernelFeatureMap(
                n_features=n_features,
                feature_map_type="zz",
                entanglement="linear"
            )
        
        elif encoding_method == "qks":
            return QuantumKitchenSinks(
                n_features=n_features,
                n_random_circuits=50,
                n_qubits=n_qubits
            )
        
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")


class Trainer:
    """Main training class."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_time': []
        }
    
    def setup_model_and_optimizer(self, n_features: int, n_classes: int):
        """Initialize model and optimizer."""
        self.model = ModelFactory.create_model(
            self.config.encoding_method, 
            n_features, 
            n_classes,
            self.config.n_layers
        )
        
        if self.config.encoding_method not in ["kernel", "qks"]:
            self.optimizer = AdamOptimizer(stepsize=self.config.learning_rate)
    
    def train_quantum_model(self, x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train quantum neural network model."""
        # Initialize weights based on model type
        if hasattr(self.model, 'initialize_weights'):
            weights = self.model.initialize_weights()
        else:
            # For DRU and other custom models
            n_qubits = self.model.get('n_qubits', 4)
            n_layers = self.model.get('n_layers', 3)
            weights = np.random.normal(0, np.pi, (n_layers, n_qubits))
        
        def cost_function(w, x_batch, y_batch):
            """Cost function for optimization."""
            predictions = []
            for x, y in zip(x_batch, y_batch):
                if self.config.encoding_method == "dru":
                    pred = self.model['circuit'](x, w)
                else:
                    pred = self.model.predict(x, w)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            # Cross-entropy loss
            return -np.mean(y_batch * np.log(predictions + 1e-8) + 
                          (1 - y_batch) * np.log(1 - predictions + 1e-8))
        
        # Training loop
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Batch training
            n_batches = len(x_train) // self.config.batch_size
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Update weights
                weights, loss = self.optimizer.step_and_cost(
                    lambda w: cost_function(w, x_batch, y_batch), weights
                )
                epoch_loss += loss
            
            # Evaluation
            train_acc = self.evaluate(x_train, y_train, weights)
            val_acc = self.evaluate(x_val, y_val, weights)
            
            # Record metrics
            self.history['train_loss'].append(epoch_loss / n_batches)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(time.time() - start_time)
            
            # Learning rate decay
            self.optimizer.stepsize *= self.config.lr_decay
            
            if self.config.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Acc={train_acc:.3f}, "
                      f"Val Acc={val_acc:.3f}, Loss={epoch_loss/n_batches:.4f}")
        
        return weights
    
    def train_kernel_model(self, x_train: np.ndarray, y_train: np.ndarray,
                          x_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train kernel-based quantum model."""
        # Compute quantum kernel matrix
        K_train = self.model.compute_kernel_matrix(x_train, x_train)
        K_val = self.model.compute_kernel_matrix(x_val, x_train)
        
        # Train SVM on quantum kernel
        svm = SVC(kernel='precomputed', C=1.0)
        svm.fit(K_train, y_train)
        
        # Evaluate
        train_acc = svm.score(K_train, y_train)
        val_acc = svm.score(K_val, y_val)
        
        self.history['train_acc'] = [train_acc] * self.config.epochs
        self.history['val_acc'] = [val_acc] * self.config.epochs
        
        return svm
    
    def train_qks_model(self, x_train: np.ndarray, y_train: np.ndarray,
                       x_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train Quantum Kitchen Sinks model."""
        # Generate random quantum features
        train_features = self.model.compute_features(x_train)
        val_features = self.model.compute_features(x_val)
        
        # Train linear classifier on quantum features
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(train_features, y_train)
        
        # Evaluate
        train_acc = classifier.score(train_features, y_train)
        val_acc = classifier.score(val_features, y_val)
        
        self.history['train_acc'] = [train_acc] * self.config.epochs
        self.history['val_acc'] = [val_acc] * self.config.epochs
        
        return classifier
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> float:
        """Evaluate model accuracy."""
        predictions = []
        
        for x_sample in x:
            if self.config.encoding_method == "dru":
                pred = self.model['circuit'](x_sample, weights)
                pred = 1 if pred > 0 else 0
            elif hasattr(self.model, 'predict'):
                pred = self.model.predict(x_sample, weights)
                pred = np.argmax(pred) if len(pred.shape) > 0 else (1 if pred > 0.5 else 0)
            predictions.append(pred)
        
        return accuracy_score(y, predictions)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Main training method."""
        print(f"Training {self.config.encoding_method.upper()} model...")
        print(f"Data shape: {x_train.shape}, Classes: {len(np.unique(y_train))}")
        
        if self.config.encoding_method == "kernel":
            return self.train_kernel_model(x_train, y_train, x_val, y_val)
        elif self.config.encoding_method == "qks":
            return self.train_qks_model(x_train, y_train, x_val, y_val)
        else:
            return self.train_quantum_model(x_train, y_train, x_val, y_val)


class ResultsAnalyzer:
    """Analyze and visualize training results."""
    
    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy curves
        epochs = range(len(history['train_acc']))
        axes[0].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[0].plot(epochs, history['val_acc'], 'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training Curves')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss curve (if available)
        if 'train_loss' in history and len(history['train_loss']) > 0:
            axes[1].plot(epochs, history['train_loss'], 'g-')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def generate_report(config: TrainingConfig, history: Dict, 
                       test_metrics: Dict, runtime: float) -> Dict:
        """Generate comprehensive training report."""
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset': config.dataset_name,
                'encoding_method': config.encoding_method,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'n_layers': config.n_layers
            },
            'performance': {
                'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
                'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
                'test_accuracy': test_metrics.get('accuracy', 0),
                'test_f1': test_metrics.get('f1', 0)
            },
            'computational_cost': {
                'total_runtime_seconds': runtime,
                'avg_epoch_time': np.mean(history['epoch_time']) if history['epoch_time'] else 0,
                'circuit_depth': 'N/A',  # Would need circuit analysis
                'gate_count': 'N/A'
            }
        }
        
        return report
    
    @staticmethod
    def save_report(report: Dict, filepath: str):
        """Save report as JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Quantum ML Training Script')
    parser.add_argument('--encoding', type=str, default='ae', 
                       choices=['ae', 'dru', 'amp', 'hybrid', 'kernel', 'qks'],
                       help='Encoding method')
    parser.add_argument('--dataset', type=str, default='fashion_mnist_pca32_T2',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = TrainingConfig()
    config.encoding_method = args.encoding
    config.dataset_name = args.dataset
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.n_layers = args.layers
    
    print(f"Starting quantum ML experiment with {config.encoding_method.upper()} encoding")
    print(f"Dataset: {config.dataset_name}")
    
    try:
        # Load data
        print("\n1. Loading and preprocessing data...")
        x_train, y_train, x_test, y_test = DataLoader.load_dataset(config.dataset_name)
        x_train_split, x_val, y_train_split, y_val = DataLoader.create_train_val_split(
            x_train, y_train, config.val_split
        )
        
        print(f"Train: {x_train_split.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
        
        # Setup model and trainer
        print("\n2. Setting up model and trainer...")
        trainer = Trainer(config)
        n_features = x_train_split.shape[1]
        n_classes = len(np.unique(y_train))
        trainer.setup_model_and_optimizer(n_features, n_classes)
        
        # Training
        print("\n3. Starting training...")
        start_time = time.time()
        
        trained_model = trainer.train(x_train_split, y_train_split, x_val, y_val)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluation on test set
        print("\n4. Evaluating on test set...")
        if config.encoding_method in ["kernel", "qks"]:
            if config.encoding_method == "kernel":
                K_test = trainer.model.compute_kernel_matrix(x_test, x_train_split)
                test_predictions = trained_model.predict(K_test)
            else:
                test_features = trainer.model.compute_features(x_test)
                test_predictions = trained_model.predict(test_features)
        else:
            test_predictions = []
            for x_sample in x_test:
                if config.encoding_method == "dru":
                    pred = trainer.model['circuit'](x_sample, trained_model)
                    pred = 1 if pred > 0 else 0
                else:
                    pred = trainer.model.predict(x_sample, trained_model)
                    pred = np.argmax(pred) if len(pred.shape) > 0 else (1 if pred > 0.5 else 0)
                test_predictions.append(pred)
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_predictions),
            'f1': f1_score(y_test, test_predictions, average='weighted')
        }
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
        # Results analysis and visualization
        print("\n5. Generating results and visualizations...")
        
        # Create results directory for this experiment
        exp_dir = os.path.join(RESULTS_DIR, f"{config.encoding_method}_{config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Plot training curves
        plot_path = os.path.join(exp_dir, "training_curves.png")
        ResultsAnalyzer.plot_training_curves(trainer.history, plot_path)
        
        # Generate comprehensive report
        report = ResultsAnalyzer.generate_report(config, trainer.history, test_metrics, training_time)
        report_path = os.path.join(exp_dir, "experiment_report.json")
        ResultsAnalyzer.save_report(report, report_path)
        
        # Print summary
        print(f"\n6. Experiment Summary:")
        print(f"   Method: {config.encoding_method.upper()}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Final Val Accuracy: {trainer.history['val_acc'][-1]:.4f}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Results saved to: {exp_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()