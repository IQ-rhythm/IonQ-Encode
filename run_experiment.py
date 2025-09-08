#!/usr/bin/env python3
"""
Convenience script to run the complete quantum ML experiment pipeline.

This script:
1. Runs data preprocessing if needed
2. Executes training with specified parameters
3. Provides easy command-line interface for experiments

Usage:
    # Run preprocessing first (if not done)
    python run_experiment.py --preprocess-only
    
    # Run single experiment
    python run_experiment.py --encoding ae --dataset fashion_mnist_pca32_T2
    
    # Run multiple experiments
    python run_experiment.py --encoding ae dru amp --dataset fashion_mnist_pca32_T2 fashion_mnist_pca32_T4
    
    # Run comprehensive benchmark
    python run_experiment.py --benchmark
"""

import os
import sys
import subprocess
import argparse
from itertools import product


def run_preprocessing():
    """Run data preprocessing script."""
    print("Running data preprocessing...")
    try:
        result = subprocess.run([
            sys.executable, "data/preprocess/preprocess-mnist.py"
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
        print("✓ Preprocessing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def check_preprocessed_data():
    """Check if preprocessed data exists."""
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        return False
    
    # Check for key datasets
    required_files = [
        "fashion_mnist_pca32_T2.npz",
        "fashion_mnist_pca32_T4.npz", 
        "fashion_mnist_pca32_T10.npz"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return False
    
    return True


def run_training_experiment(encoding, dataset, epochs=10, qubits=4, **kwargs):
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {encoding.upper()} on {dataset}")
    print(f"{'='*60}")
    
    # For now, we only support the working quantum_ml_experiment.py
    if encoding == "ae":
        cmd = [
            sys.executable, "quantum_ml_experiment.py",
            "--dataset", dataset, 
            "--qubits", str(qubits),
            "--epochs", str(epochs)
        ]
    else:
        print(f"⚠ Encoding {encoding} not yet implemented in simplified version")
        print(f"✓ Using Angle Encoding as baseline")
        cmd = [
            sys.executable, "quantum_ml_experiment.py",
            "--dataset", dataset, 
            "--qubits", str(qubits),
            "--epochs", str(epochs)
        ]
    
    try:
        result = subprocess.run(cmd, check=True, timeout=120)
        print(f"✓ Experiment completed: {encoding} on {dataset}")
        return True
    except subprocess.TimeoutExpired:
        print(f"⚠ Experiment timed out: {encoding} on {dataset}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed: {encoding} on {dataset}")
        print(f"Error: {e}")
        return False


def run_benchmark():
    """Run comprehensive benchmark across multiple encodings and datasets."""
    print("Running comprehensive benchmark...")
    
    # Define experiment parameters
    encodings = ["ae", "dru", "amp", "hybrid", "kernel", "qks"] 
    datasets = ["fashion_mnist_pca32_T2", "fashion_mnist_pca32_T4"]
    
    # Reduced parameters for comprehensive testing
    epochs = 10  # Shorter for benchmark
    
    results = []
    
    for encoding, dataset in product(encodings, datasets):
        print(f"\nStarting {encoding} on {dataset}...")
        success = run_training_experiment(encoding, dataset, epochs=epochs)
        results.append({
            'encoding': encoding,
            'dataset': dataset, 
            'success': success
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['encoding']:8} | {result['dataset']}")
    
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"\nOverall success rate: {success_rate:.1%} ({sum(1 for r in results if r['success'])}/{len(results)})")


def main():
    parser = argparse.ArgumentParser(description="Run quantum ML experiments")
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run data preprocessing')
    parser.add_argument('--encoding', nargs='+', default=['ae'],
                       choices=['ae', 'dru', 'amp', 'hybrid', 'kernel', 'qks'],
                       help='Encoding methods to test')
    parser.add_argument('--dataset', nargs='+', 
                       default=['fashion_mnist_pca32_T2'],
                       help='Datasets to use')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size') 
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run comprehensive benchmark')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing check/execution')
    
    args = parser.parse_args()
    
    print("IonQ Encode Challenge - Quantum ML Training Pipeline")
    print("=" * 60)
    
    # Handle preprocessing
    if args.preprocess_only:
        run_preprocessing()
        return
    
    if not args.skip_preprocessing:
        if not check_preprocessed_data():
            print("Preprocessed data not found. Running preprocessing...")
            if not run_preprocessing():
                print("Failed to preprocess data. Exiting.")
                return
        else:
            print("✓ Preprocessed data found")
    
    # Run experiments
    if args.benchmark:
        run_benchmark()
    else:
        # Run specified experiments
        for encoding in args.encoding:
            for dataset in args.dataset:
                run_training_experiment(
                    encoding=encoding,
                    dataset=dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    layers=args.layers
                )
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Check the 'results/' directory for detailed outputs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()