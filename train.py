#!/usr/bin/env python3
"""
Quantum Machine Learning Training Script

A clean, efficient training script that supports all encoding methods:
- Angle Encoding
- Amplitude Encoding (exact and approximate)
- Hybrid Encoding
- Kernel Feature Maps
- Quantum Kitchen Sinks (QKS)

Configuration is loaded from YAML files to separate variables from code.
"""

import os
import sys
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.optim as optim
import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import quantum encodings
from quantum_encodings import (
    AngleEncodingClassifier,
    AmplitudeEncodingClassifier,
    HybridEncodingClassifier,
    QuantumKernelFeatureMap,
    QuantumKitchenSinks
)


class QuantumMLTrainer:
    """Clean, efficient trainer for quantum machine learning experiments."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed dataset."""
        data_dir = self.config['data']['processed_dir']
        path = os.path.join(data_dir, f"{dataset_name}.npz")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
            
        data = np.load(path)
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']
    
    def preprocess_features(self, x: np.ndarray, target_dim: Optional[int] = None) -> np.ndarray:
        """Reduce feature dimensionality using variance-based selection."""
        if target_dim is None:
            target_dim = self.config['circuit']['target_feature_dim']
            
        if x.shape[1] > target_dim:
            variances = np.var(x, axis=0)
            top_indices = np.argsort(variances)[-target_dim:]
            return x[:, top_indices]
        return x
    
    def create_model(self, encoding_type: str, n_features: int, **kwargs) -> Any:
        """Create quantum model based on encoding type."""
        default_qubits = self.config.get('circuit', {}).get('default_n_qubits', 4)
        n_qubits = kwargs.get('n_qubits') or min(n_features, default_qubits)
        n_layers = kwargs.get('n_layers') or self.config.get('circuit', {}).get('default_n_layers', 2)
        
        if encoding_type == 'angle':
            return AngleEncodingClassifier(
                n_features=n_qubits,
                n_layers=n_layers,
                seed=self.config['data']['random_state']
            )
        elif encoding_type == 'amplitude':
            method = kwargs.get('method', 'exact')
            return AmplitudeEncodingClassifier(
                n_features=n_features,
                n_layers=n_layers,
                method=method,
                seed=self.config['data']['random_state']
            )
        elif encoding_type == 'hybrid':
            return HybridEncodingClassifier(
                n_features=n_features,
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=self.config['data']['random_state']
            )
        elif encoding_type == 'kernel':
            map_type = kwargs.get('map_type', 'zz')
            return QuantumKernelFeatureMap(
                n_features=n_features,
                map_type=map_type,
                repetitions=kwargs.get('repetitions', 2)
            )
        elif encoding_type == 'qks':
            return QuantumKitchenSinks(
                n_features=kwargs.get('qks_features', 50),
                n_qubits=n_qubits,
                n_layers=n_layers,
                seed=self.config['data']['random_state']
            )
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def train_model(self, model: Any, x_train: np.ndarray, y_train: np.ndarray, 
                   x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, list]:
        """Train quantum model with specified data."""
        # Convert to torch tensors
        x_train_torch = torch.tensor(x_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.float32)
        x_val_torch = torch.tensor(x_val, dtype=torch.float32)
        y_val_torch = torch.tensor(y_val, dtype=torch.float32)
        
        # Setup PyTorch optimizer
        optimizer = optim.Adam([model.weights], lr=self.config['optimizer']['learning_rate'])
        
        # Training configuration
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        max_batches = self.config['training']['max_batches_per_epoch']
        eval_freq = self.config['training']['evaluation_frequency']
        train_eval_samples = self.config['training']['train_eval_samples']
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        def compute_loss(x_batch, y_batch):
            """Binary cross-entropy loss function."""
            predictions = []
            
            for x_sample in x_batch:
                pred_raw = model(x_sample)
                # Convert to tensor if needed
                if not isinstance(pred_raw, torch.Tensor):
                    pred_raw = torch.tensor(pred_raw, dtype=torch.float32)
                pred = torch.sigmoid(pred_raw)
                predictions.append(pred)
            
            predictions = torch.stack(predictions)
            predictions = torch.clamp(predictions, 1e-8, 1 - 1e-8)
            
            loss = -torch.mean(y_batch * torch.log(predictions) + 
                              (1 - y_batch) * torch.log(1 - predictions))
            return loss
        
        def evaluate_accuracy(model, x, y):
            """Evaluate model accuracy."""
            predictions = []
            for x_sample in x:
                pred_raw = model(x_sample)
                # Convert to float if it's a tensor
                if hasattr(pred_raw, 'item'):
                    pred_raw = pred_raw.item()
                pred = 1 if pred_raw > 0 else 0
                predictions.append(pred)
            return accuracy_score(y.numpy(), predictions)
        
        # Training loop
        n_batches = len(x_train_torch) // batch_size
        
        print(f"Training for {epochs} epochs with {min(n_batches, max_batches)} batches per epoch")
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_processed_batches = min(n_batches, max_batches)
            
            for i in range(n_processed_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(x_train_torch))
                
                x_batch = x_train_torch[start_idx:end_idx]
                y_batch = y_train_torch[start_idx:end_idx]
                
                optimizer.zero_grad()
                loss = compute_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Evaluation
            if epoch % eval_freq == 0:
                # Limit training evaluation for speed
                eval_indices = np.random.choice(
                    len(x_train_torch), 
                    min(train_eval_samples, len(x_train_torch)), 
                    replace=False
                )
                x_train_eval = x_train_torch[eval_indices]
                y_train_eval = y_train_torch[eval_indices]
                
                train_acc = evaluate_accuracy(model, x_train_eval, y_train_eval)
                val_acc = evaluate_accuracy(model, x_val_torch, y_val_torch)
                
                avg_loss = epoch_loss / n_processed_batches
                history['train_loss'].append(avg_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        return history
    
    def evaluate_model(self, model: Any, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate final model performance on test set."""
        x_test_torch = torch.tensor(x_test, dtype=torch.float32)
        
        predictions = []
        for x_sample in x_test_torch:
            pred_raw = model(x_sample)
            # Convert to float if it's a tensor
            if hasattr(pred_raw, 'item'):
                pred_raw = pred_raw.item()
            pred = 1 if pred_raw > 0 else 0
            predictions.append(pred)
        
        return accuracy_score(y_test, predictions)
    
    def save_results(self, experiment_name: str, model_info: dict, 
                    history: dict, test_acc: float, training_time: float):
        """Save experiment results and visualizations."""
        if not (self.config['output']['save_plots'] or self.config['output']['save_reports']):
            return
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = self.results_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save training curves
        if self.config['output']['save_plots'] and history['train_acc']:
            plt.figure(figsize=(12, 4))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(history['train_acc'], 'b-o', label='Train', markersize=4)
            plt.plot(history['val_acc'], 'r-o', label='Validation', markersize=4)
            plt.xlabel('Evaluation Step')
            plt.ylabel('Accuracy')
            plt.title(f'Training Curves - Test Acc: {test_acc:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(history['train_loss'], 'g-o', markersize=4)
            plt.xlabel('Evaluation Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(exp_dir / "training_curves.png", 
                       dpi=self.config['output']['plot_dpi'], bbox_inches='tight')
            plt.close()
        
        # Save experiment report
        if self.config['output']['save_reports']:
            report = {
                'experiment': {
                    'name': experiment_name,
                    'timestamp': timestamp,
                    **model_info
                },
                'results': {
                    'test_accuracy': float(test_acc),
                    'final_train_acc': float(history['train_acc'][-1]) if history['train_acc'] else 0,
                    'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0,
                    'training_time': float(training_time)
                },
                'config': self.config
            }
            
            import json
            with open(exp_dir / "report.json", 'w') as f:
                json.dump(report, f, indent=2)
        
        print(f"Results saved to: {exp_dir}")
    
    def run_experiment(self, encoding_type: str, dataset_name: Optional[str] = None, 
                      n_qubits: Optional[int] = None, epochs: Optional[int] = None, 
                      **kwargs) -> Tuple[float, float]:
        """Run a complete quantum ML experiment."""
        # Use provided parameters or defaults from config
        dataset_name = dataset_name or self.config['data']['default_dataset']
        if epochs is not None:
            self.config['training']['epochs'] = epochs
        
        print(f"Running Quantum ML Experiment: {encoding_type.upper()}")
        print(f"Dataset: {dataset_name}")
        print(f"Encoding: {encoding_type}")
        print("=" * 50)
        
        # Load and preprocess data
        print("1. Loading and preprocessing data...")
        x_train, y_train, x_test, y_test = self.load_dataset(dataset_name)
        
        # For Fashion-MNIST, apply dimensionality reduction for encodings that need it
        default_qubits = self.config.get('circuit', {}).get('default_n_qubits', 4)
        if encoding_type in ['angle', 'hybrid']:
            target_dim = n_qubits or default_qubits
            if x_train.shape[1] > target_dim:
                print(f"   Reducing features from {x_train.shape[1]} to {target_dim} for {encoding_type} encoding")
                x_train = self.preprocess_features(x_train, target_dim)
                x_test = self.preprocess_features(x_test, target_dim)
            else:
                target_dim = x_train.shape[1]
        
        # Create train/validation split
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, 
            test_size=self.config['data']['validation_split'],
            random_state=self.config['data']['random_state'],
            stratify=y_train if self.config['data']['stratify'] else None
        )
        
        print(f"Train: {x_train_split.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
        
        # Create model
        print("2. Creating quantum model...")
        model = self.create_model(
            encoding_type, 
            n_features=x_train_split.shape[1],
            n_qubits=n_qubits,
            **kwargs
        )
        
        model_info = model.get_circuit_info() if hasattr(model, 'get_circuit_info') else {
            'encoding_type': encoding_type,
            'n_features': x_train_split.shape[1]
        }
        print(f"Model: {model_info}")
        
        # Training
        print("3. Training model...")
        start_time = time.time()
        history = self.train_model(model, x_train_split, y_train_split, x_val, y_val)
        training_time = time.time() - start_time
        
        # Final evaluation
        print("4. Final evaluation...")
        test_acc = self.evaluate_model(model, x_test, y_test)
        
        # Results
        print("5. Results:")
        print(f"   Final Test Accuracy: {test_acc:.4f}")
        print(f"   Training Time: {training_time:.1f} seconds")
        
        # Save results
        experiment_name = f"{encoding_type}_{dataset_name}"
        self.save_results(experiment_name, model_info, history, test_acc, training_time)
        
        return test_acc, training_time


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum ML Training Script")
    parser.add_argument('--encoding', default='angle', 
                       choices=['angle', 'amplitude', 'hybrid', 'kernel', 'qks'],
                       help='Encoding method to use')
    parser.add_argument('--dataset', default=None,
                       help='Dataset name (default from config)')
    parser.add_argument('--qubits', type=int, default=None,
                       help='Number of qubits (default from config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default from config)')
    parser.add_argument('--config', default='config/training_config.yaml',
                       help='Path to configuration file')
    
    # Encoding-specific arguments
    parser.add_argument('--amplitude-method', default='exact',
                       choices=['exact', 'approximate'],
                       help='Amplitude encoding method')
    parser.add_argument('--kernel-type', default='zz',
                       choices=['zz', 'iqp'],
                       help='Kernel feature map type')
    parser.add_argument('--qks-features', type=int, default=50,
                       help='Number of QKS features')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = QuantumMLTrainer(args.config)
    
    # Prepare encoding-specific kwargs
    kwargs = {}
    if args.encoding == 'amplitude':
        kwargs['method'] = args.amplitude_method
    elif args.encoding == 'kernel':
        kwargs['map_type'] = args.kernel_type
    elif args.encoding == 'qks':
        kwargs['qks_features'] = args.qks_features
    
    # Run experiment
    try:
        test_acc, training_time = trainer.run_experiment(
            encoding_type=args.encoding,
            dataset_name=args.dataset,
            n_qubits=args.qubits,
            epochs=args.epochs,
            **kwargs
        )
        print(f"\nExperiment completed successfully!")
        print(f"Final accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()