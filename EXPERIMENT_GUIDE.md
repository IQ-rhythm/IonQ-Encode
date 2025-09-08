# Quantum ML Training Pipeline - Experiment Guide

This guide explains how to use the complete training pipeline for the IonQ Encode Challenge.

## Quick Start

### 1. Run Data Preprocessing
```bash
python run_experiment.py --preprocess-only
```

### 2. Run a Simple Experiment
```bash
# Angle Encoding on binary classification
python run_experiment.py --encoding ae --dataset fashion_mnist_pca32_T2 --epochs 5

# Direct usage of quantum ML experiment
python quantum_ml_experiment.py --dataset fashion_mnist_pca32_T2 --qubits 4 --epochs 8

# 4-class classification with more qubits
python quantum_ml_experiment.py --dataset fashion_mnist_pca32_T4 --qubits 6 --epochs 10
```

### 3. Run Multiple Configurations  
```bash
# Compare different qubit counts
python quantum_ml_experiment.py --comparison
```

## Available Encoding Methods

| Method | Code | Description | Best Use Case |
|--------|------|-------------|---------------|
| Angle Encoding | `ae` | RY rotations, shallow circuits | Low-dimensional baseline |
| Data Re-uploading | `dru` | Multi-layer data injection | Mid-scale expressivity |
| Amplitude Encoding | `amp` | Direct state preparation | High-dimensional compression |
| Hybrid Encoding | `hybrid` | Combined angle + amplitude | Multi-class balanced |
| Kernel Feature Map | `kernel` | ZZ/IQP + SVM | Robust baseline |
| Quantum Kitchen Sinks | `qks` | Random circuits + linear | Quick baseline |

## Available Datasets

| Dataset | Classes | Features | Description |
|---------|---------|----------|-------------|
| `fashion_mnist_pca32_T2` | 2 | 32 | Binary: T-shirt vs Pullover |
| `fashion_mnist_pca32_T4` | 4 | 32 | 4-class: T-shirt, Pullover, Coat, Shirt |
| `fashion_mnist_pca32_T10` | 10 | 32 | Full Fashion-MNIST (reduced) |
| `fashion_mnist_pca64_T2` | 2 | 64 | Higher dimensional binary |
| `fashion_mnist_pca64_T4` | 4 | 64 | Higher dimensional 4-class |

## Example Experiments

### Noiseless Environment Testing (Week 3)
```bash
# Test all encodings on binary classification
python run_experiment.py --encoding ae dru amp hybrid kernel qks --dataset fashion_mnist_pca32_T2 --epochs 20

# Compare PCA dimensions
python run_experiment.py --encoding ae --dataset fashion_mnist_pca32_T2 fashion_mnist_pca64_T2 --epochs 15

# Multi-class evaluation
python run_experiment.py --encoding hybrid --dataset fashion_mnist_pca32_T4 --epochs 25
```

### Resource Analysis
```bash
# Deep vs shallow circuits
python run_experiment.py --encoding dru --dataset fashion_mnist_pca32_T2 --layers 1 --epochs 20
python run_experiment.py --encoding dru --dataset fashion_mnist_pca32_T2 --layers 5 --epochs 20

# Learning rate sensitivity
python run_experiment.py --encoding ae --dataset fashion_mnist_pca32_T2 --lr 1e-3 --epochs 30
python run_experiment.py --encoding ae --dataset fashion_mnist_pca32_T2 --lr 1e-2 --epochs 30
```

## Results and Analysis

Results are automatically saved to `results/` with timestamped directories:

```
results/
├── ae_fashion_mnist_pca32_T2_20250908_143022/
│   ├── experiment_report.json      # Detailed metrics
│   └── training_curves.png         # Visualization
└── dru_fashion_mnist_pca32_T4_20250908_143156/
    ├── experiment_report.json
    └── training_curves.png
```

### Key Metrics Tracked
- **Accuracy**: Train, validation, and test accuracy
- **F1 Score**: Weighted F1 for multi-class
- **Training Time**: Total and per-epoch timing
- **Circuit Properties**: Depth and gate count (when available)

## Advanced Usage

### Direct Training Script
```bash
python training_script.py --encoding amp --dataset fashion_mnist_pca64_T4 --epochs 30 --batch_size 16 --lr 3e-3
```

### Custom Hyperparameters
```bash
python run_experiment.py \
    --encoding dru \
    --dataset fashion_mnist_pca32_T4 \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-2 \
    --layers 4
```

## Implementation Notes

### Data Flow
1. **Download**: Fashion-MNIST via torchvision
2. **Resize**: 28×28 → 8×8 (64D)  
3. **PCA**: Reduce to 16/32/64 dimensions
4. **Normalize**: StandardScaler + [0,1] scaling
5. **Split**: 80% train, 10% val, 10% test
6. **Encode**: Apply quantum encoding
7. **Train**: Adam optimizer with learning rate decay
8. **Evaluate**: Accuracy and F1 on test set

### Circuit Design Principles
- **Angle Encoding**: Single RY per feature + entangling layers
- **Data Re-uploading**: Repeated data injection across layers
- **Amplitude Encoding**: Exact (≤32D) or approximate (>32D)
- **Hybrid**: Combines angle and amplitude encoding
- **Kernel**: Fixed feature maps with classical SVM
- **QKS**: Random quantum feature extraction

## Troubleshooting

### Common Issues
1. **Missing preprocessed data**: Run `--preprocess-only` first
2. **Out of memory**: Reduce `--batch_size` or use smaller dataset
3. **Slow convergence**: Increase `--lr` or `--epochs`
4. **Import errors**: Check dependencies in `requirements.txt`

### Performance Tips
- Use binary classification (`T2`) for initial testing
- Start with shallow circuits (`--layers 2`)
- Use smaller batch sizes for memory efficiency
- Enable `--skip-preprocessing` for repeated runs

## Next Steps

### Week 4: Noise Modeling
After establishing noiseless baselines, you can extend the pipeline for:
- Noise model integration (depolarizing, readout error)
- Shot noise simulation (finite shots)
- Error mitigation techniques
- Hardware submission preparation

The modular design allows easy extension for noisy simulations and hardware experiments.