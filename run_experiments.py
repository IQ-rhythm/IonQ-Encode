#!/usr/bin/env python3
"""
Run multiple quantum machine learning experiments with different configurations.

This script demonstrates how to use the training system with various
encoding methods and Fashion-MNIST dataset variations.
"""

from train import QuantumMLTrainer
import time

def main():
    """Run a series of experiments with different configurations."""
    
    # Initialize trainer
    trainer = QuantumMLTrainer("config/training_config.yaml")
    
    # Define all encoding methods and datasets to test
    encodings = ["angle", "amplitude", "hybrid", "kernel", "qks"]
    datasets = [
        "fashion_mnist_pca16_T2", "fashion_mnist_pca32_T2", "fashion_mnist_pca64_T2",
        "fashion_mnist_pca16_T4", "fashion_mnist_pca32_T4", "fashion_mnist_pca64_T4",
        "fashion_mnist_pca16_T10", "fashion_mnist_pca32_T10", "fashion_mnist_pca64_T10"
    ]
    
    experiments = []
    
    # Generate all combinations
    for encoding in encodings:
        for dataset in datasets:
            # Configure based on encoding and dataset
            exp_config = {
                "name": f"{encoding.upper()} - {dataset}",
                "encoding": encoding,
                "dataset": dataset,
                "epochs": 3  # Short epochs for comprehensive testing
            }
            
            # Encoding-specific configurations
            if encoding == "angle":
                exp_config["n_qubits"] = 4 if "T2" in dataset else (6 if "T4" in dataset else 8)
            elif encoding == "amplitude":
                exp_config["method"] = "exact"
                exp_config["epochs"] = 2  # Amplitude encoding is slower
            elif encoding == "hybrid":
                exp_config["n_qubits"] = 4 if "T2" in dataset else (6 if "T4" in dataset else 8)
            elif encoding == "kernel":
                exp_config["map_type"] = "zz"
                exp_config["repetitions"] = 2
            elif encoding == "qks":
                exp_config["qks_features"] = 32 if "pca32" in dataset else (16 if "pca16" in dataset else 64)
                exp_config["n_qubits"] = 4
            
            experiments.append(exp_config)
    
    results = []
    total_start_time = time.time()
    
    print("Starting Quantum ML Experiment Suite")
    print("=" * 60)
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_config['name']}")
        print("-" * 40)
        
        try:
            # Extract parameters
            encoding = exp_config.pop('encoding')
            dataset = exp_config.pop('dataset') 
            name = exp_config.pop('name')
            
            # Run experiment
            test_acc, training_time = trainer.run_experiment(
                encoding_type=encoding,
                dataset_name=dataset,
                **exp_config
            )
            
            results.append({
                'name': name,
                'encoding': encoding,
                'dataset': dataset,
                'test_accuracy': test_acc,
                'training_time': training_time,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            results.append({
                'name': exp_config.get('name', 'Unknown'),
                'encoding': exp_config.get('encoding', 'Unknown'),
                'dataset': exp_config.get('dataset', 'Unknown'),
                'test_accuracy': 0.0,
                'training_time': 0.0,
                'status': f'failed: {str(e)}'
            })
    
    # Print summary
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print()
    
    # Print results organized by encoding method
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Successful experiments: {success_count}/{len(experiments)}")
    print()
    
    for encoding in encodings:
        print(f"{encoding.upper()} ENCODING:")
        encoding_results = [r for r in results if r['encoding'] == encoding]
        
        for result in encoding_results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            dataset_short = result['dataset'].replace('fashion_mnist_', '')
            print(f"  {status_symbol} {dataset_short:12} | Acc: {result['test_accuracy']:.3f} | Time: {result['training_time']:4.1f}s")
        print()
    
    print(f"All results saved in results/ directory")
    print("Each experiment has its own timestamped folder with detailed reports and plots")

if __name__ == "__main__":
    main()