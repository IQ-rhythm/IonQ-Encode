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
    
    # Define experiment configurations
    experiments = [
        # Binary classification experiments (T2)
        {
            "name": "Angle Encoding - Binary (PCA32)",
            "encoding": "angle",
            "dataset": "fashion_mnist_pca32_T2",
            "n_qubits": 4,
            "epochs": 5
        },
        {
            "name": "Amplitude Encoding - Binary (PCA16)", 
            "encoding": "amplitude",
            "dataset": "fashion_mnist_pca16_T2",
            "method": "exact",
            "epochs": 3
        },
        {
            "name": "Angle Encoding - 4-class (PCA32)",
            "encoding": "angle", 
            "dataset": "fashion_mnist_pca32_T4",
            "n_qubits": 6,
            "epochs": 5
        }
    ]
    
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
    
    for result in results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} {result['name']:35} | "
              f"Acc: {result['test_accuracy']:.3f} | "
              f"Time: {result['training_time']:4.1f}s | "
              f"{result['dataset']}")
    
    print(f"\nResults saved in results/ directory")
    print("Each experiment has its own timestamped folder with:")
    print("- report.json: Detailed experiment metadata and results")
    print("- training_curves.png: Training and validation accuracy plots")

if __name__ == "__main__":
    main()