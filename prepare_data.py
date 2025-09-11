#!/usr/bin/env python3
"""
Data preparation script for Fashion-MNIST experiments.

This script runs the preprocessing pipeline to create PCA-reduced 
Fashion-MNIST datasets for quantum machine learning experiments.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_datasets_exist():
    """Check if preprocessed datasets already exist."""
    data_dir = Path("data/processed")
    
    expected_files = [
        "fashion_mnist_pca16_T2.npz",
        "fashion_mnist_pca32_T2.npz", 
        "fashion_mnist_pca64_T2.npz",
        "fashion_mnist_pca16_T4.npz",
        "fashion_mnist_pca32_T4.npz",
        "fashion_mnist_pca64_T4.npz",
        "fashion_mnist_pca16_T10.npz",
        "fashion_mnist_pca32_T10.npz",
        "fashion_mnist_pca64_T10.npz"
    ]
    
    missing_files = []
    for filename in expected_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files

def run_preprocessing():
    """Run the Fashion-MNIST preprocessing script."""
    preprocess_script = "data/preprocess/preprocess-mnist.py"
    
    if not os.path.exists(preprocess_script):
        print(f"Error: Preprocessing script not found at {preprocess_script}")
        return False
    
    print("Running Fashion-MNIST preprocessing...")
    print("This will download Fashion-MNIST dataset and create PCA-reduced versions...")
    
    try:
        result = subprocess.run([sys.executable, preprocess_script], 
                              capture_output=True, text=True, check=True)
        print("Preprocessing completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    print("Fashion-MNIST Data Preparation")
    print("=" * 40)
    
    # Check if datasets already exist
    datasets_exist, missing_files = check_datasets_exist()
    
    if datasets_exist:
        print("✓ All Fashion-MNIST datasets already exist in data/processed/")
        print("\nAvailable datasets:")
        print("• Binary classification (T2): fashion_mnist_pca{16,32,64}_T2")
        print("• 4-class classification (T4): fashion_mnist_pca{16,32,64}_T4") 
        print("• 10-class classification (T10): fashion_mnist_pca{16,32,64}_T10")
        print("\nYou can now run training experiments!")
        return True
    else:
        print(f"Missing {len(missing_files)} dataset files:")
        for file in missing_files[:5]:  # Show first 5
            print(f"  • {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        print()
        
        response = input("Would you like to run preprocessing now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return run_preprocessing()
        else:
            print("Please run preprocessing before training experiments.")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)