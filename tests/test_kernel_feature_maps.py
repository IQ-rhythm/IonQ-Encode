"""
Unit tests for kernel feature maps and quantum kitchen sinks implementations.

This module tests the functionality of ZZFeatureMap, IQPFeatureMap, and QKS
implementations with PCA-16 dataset testing.
"""

import pytest
import torch
import numpy as np
import pennylane as qml
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_encodings.kernel_feature_map import (
    validate_feature_map_params,
    build_zz_feature_map,
    build_iqp_feature_map,
    compute_kernel_matrix,
    get_feature_map_complexity,
    QuantumKernelFeatureMap,
    create_feature_map_model
)

from quantum_encodings.qks import (
    generate_random_parameters,
    build_qks_circuit,
    compute_qks_features,
    compute_qks_kernel_approximation,
    benchmark_qks_complexity,
    QuantumKitchenSinks,
    EnsembleQKS
)


class TestFeatureMapValidation:
    """Test suite for feature map validation."""
    
    def test_validate_feature_map_params_valid(self):
        """Test validation with valid parameters."""
        validate_feature_map_params(4, 3, 2)  # Should not raise
        validate_feature_map_params(8, 8, 1)  # Should not raise
        validate_feature_map_params(2, 1, 3)  # Should not raise
    
    def test_validate_feature_map_params_invalid(self):
        """Test validation with invalid parameters."""
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            validate_feature_map_params(0, 3, 2)
        
        with pytest.raises(ValueError, match="n_features must be positive"):
            validate_feature_map_params(4, 0, 2)
        
        with pytest.raises(ValueError, match="repetitions must be positive"):
            validate_feature_map_params(4, 3, 0)
        
        with pytest.raises(ValueError, match="n_features cannot exceed n_qubits"):
            validate_feature_map_params(3, 5, 2)


class TestZZFeatureMap:
    """Test suite for ZZ feature map."""
    
    def test_build_zz_feature_map_basic(self):
        """Test basic ZZ feature map construction."""
        n_qubits, n_features = 4, 3
        circuit = build_zz_feature_map(n_qubits, n_features, repetitions=2)
        
        assert callable(circuit)
        
        # Test with sample features
        features = torch.tensor([0.5, 1.0, 1.5])
        result = circuit(features)
        
        assert isinstance(result, list)
        assert len(result) == n_qubits
        
        # All expectation values should be in [-1, 1]
        for exp_val in result:
            assert -1 <= exp_val <= 1
    
    def test_build_zz_feature_map_different_entanglements(self):
        """Test ZZ feature map with different entanglement patterns."""
        n_qubits, n_features = 4, 4
        entanglement_patterns = ["linear", "circular", "full"]
        features = torch.randn(n_features)
        
        for pattern in entanglement_patterns:
            circuit = build_zz_feature_map(n_qubits, n_features, 
                                          entanglement=pattern)
            result = circuit(features)
            
            assert len(result) == n_qubits
            for exp_val in result:
                assert -1 <= exp_val <= 1
    
    def test_zz_feature_map_feature_padding_truncation(self):
        """Test ZZ feature map with different feature lengths."""
        n_qubits, n_features = 4, 3
        circuit = build_zz_feature_map(n_qubits, n_features)
        
        # Test with fewer features (should pad)
        short_features = torch.tensor([1.0, 2.0])
        result_short = circuit(short_features)
        assert len(result_short) == n_qubits
        
        # Test with more features (should truncate)
        long_features = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result_long = circuit(long_features)
        assert len(result_long) == n_qubits


class TestIQPFeatureMap:
    """Test suite for IQP feature map."""
    
    def test_build_iqp_feature_map_basic(self):
        """Test basic IQP feature map construction."""
        n_qubits, n_features = 4, 3
        circuit = build_iqp_feature_map(n_qubits, n_features, repetitions=1)
        
        assert callable(circuit)
        
        features = torch.tensor([0.5, 1.0, 1.5])
        result = circuit(features)
        
        assert isinstance(result, list)
        assert len(result) == n_qubits
        
        for exp_val in result:
            assert -1 <= exp_val <= 1
    
    def test_iqp_feature_map_repetitions(self):
        """Test IQP feature map with multiple repetitions."""
        n_qubits, n_features = 3, 3
        features = torch.tensor([0.1, 0.2, 0.3])
        
        # Test different repetitions
        for reps in [1, 2, 3]:
            circuit = build_iqp_feature_map(n_qubits, n_features, reps)
            result = circuit(features)
            
            assert len(result) == n_qubits
            for exp_val in result:
                assert -1 <= exp_val <= 1
    
    def test_iqp_feature_map_polynomial_interactions(self):
        """Test that IQP creates polynomial interactions."""
        n_qubits, n_features = 3, 3
        circuit = build_iqp_feature_map(n_qubits, n_features)
        
        # Test with orthogonal features
        features1 = torch.tensor([1.0, 0.0, 0.0])
        features2 = torch.tensor([0.0, 1.0, 0.0])
        
        result1 = circuit(features1)
        result2 = circuit(features2)
        
        # Results should be different due to polynomial interactions
        assert not torch.allclose(torch.tensor(result1), torch.tensor(result2))


class TestKernelMatrixComputation:
    """Test suite for kernel matrix computation."""
    
    def test_compute_kernel_matrix_symmetric(self):
        """Test symmetric kernel matrix computation."""
        n_qubits, n_features = 3, 2
        circuit = build_zz_feature_map(n_qubits, n_features)
        
        # Small dataset
        X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        kernel_matrix = compute_kernel_matrix(circuit, X)
        
        # Should be symmetric
        assert kernel_matrix.shape == (3, 3)
        assert torch.allclose(kernel_matrix, kernel_matrix.T, atol=1e-6)
        
        # Diagonal should be close to 1 (self-similarity)
        diagonal = torch.diag(kernel_matrix)
        assert torch.all(diagonal >= 0.5)  # Reasonable self-similarity
    
    def test_compute_kernel_matrix_asymmetric(self):
        """Test asymmetric kernel matrix computation."""
        n_qubits, n_features = 3, 2
        circuit = build_iqp_feature_map(n_qubits, n_features)
        
        X = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        Y = torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
        
        kernel_matrix = compute_kernel_matrix(circuit, X, Y)
        
        assert kernel_matrix.shape == (2, 3)
        
        # All values should be reasonable
        assert torch.all(kernel_matrix >= -1.1)  # Allow small numerical errors
        assert torch.all(kernel_matrix <= 1.1)
    
    def test_kernel_matrix_normalization(self):
        """Test kernel matrix with and without normalization."""
        n_qubits, n_features = 2, 2
        circuit = build_zz_feature_map(n_qubits, n_features)
        
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # With normalization
        kernel_norm = compute_kernel_matrix(circuit, X, normalize=True)
        
        # Without normalization
        kernel_no_norm = compute_kernel_matrix(circuit, X, normalize=False)
        
        # Should be different
        assert not torch.allclose(kernel_norm, kernel_no_norm, atol=1e-6)


class TestComplexityAnalysis:
    """Test suite for complexity analysis."""
    
    def test_get_feature_map_complexity_zz(self):
        """Test complexity analysis for ZZ feature map."""
        complexity = get_feature_map_complexity("zz", n_qubits=4, n_features=3, 
                                               repetitions=2, entanglement="linear")
        
        expected_keys = [
            "n_qubits", "n_features", "repetitions", "total_gates",
            "hadamard_gates", "rz_gates", "cnot_gates", "circuit_depth"
        ]
        
        for key in expected_keys:
            assert key in complexity
            assert isinstance(complexity[key], int)
            assert complexity[key] >= 0
        
        # Basic checks
        assert complexity["n_qubits"] == 4
        assert complexity["n_features"] == 3
        assert complexity["hadamard_gates"] == 4  # One per qubit
        assert complexity["total_gates"] > 0
    
    def test_get_feature_map_complexity_iqp(self):
        """Test complexity analysis for IQP feature map."""
        complexity = get_feature_map_complexity("iqp", n_qubits=3, n_features=3, 
                                               repetitions=1)
        
        assert complexity["n_qubits"] == 3
        assert complexity["n_features"] == 3
        assert complexity["hadamard_gates"] == 3
        assert complexity["total_gates"] > complexity["hadamard_gates"]
    
    def test_complexity_scaling_with_parameters(self):
        """Test that complexity scales appropriately with parameters."""
        # Test scaling with repetitions
        complexity1 = get_feature_map_complexity("zz", 3, 3, repetitions=1)
        complexity2 = get_feature_map_complexity("zz", 3, 3, repetitions=2)
        
        assert complexity2["total_gates"] > complexity1["total_gates"]
        
        # Test scaling with entanglement
        complexity_linear = get_feature_map_complexity("zz", 4, 4, entanglement="linear")
        complexity_full = get_feature_map_complexity("zz", 4, 4, entanglement="full")
        
        assert complexity_full["total_gates"] >= complexity_linear["total_gates"]


class TestQuantumKernelFeatureMap:
    """Test suite for QuantumKernelFeatureMap wrapper class."""
    
    def test_initialization_zz(self):
        """Test ZZ feature map initialization."""
        feature_map = QuantumKernelFeatureMap("zz", n_qubits=4, n_features=3, 
                                             repetitions=2, seed=42)
        
        assert feature_map.feature_map_type == "zz"
        assert feature_map.n_qubits == 4
        assert feature_map.n_features == 3
        assert callable(feature_map.circuit)
        assert isinstance(feature_map.complexity, dict)
    
    def test_initialization_iqp(self):
        """Test IQP feature map initialization."""
        feature_map = QuantumKernelFeatureMap("iqp", n_qubits=3, n_features=3, seed=42)
        
        assert feature_map.feature_map_type == "iqp"
        assert callable(feature_map.circuit)
    
    def test_invalid_feature_map_type(self):
        """Test initialization with invalid feature map type."""
        with pytest.raises(ValueError, match="Unknown feature map type"):
            QuantumKernelFeatureMap("invalid", 3, 3)
    
    def test_callable_interface(self):
        """Test callable interface."""
        feature_map = QuantumKernelFeatureMap("zz", 3, 2, seed=42)
        features = torch.tensor([0.5, 1.0])
        
        result1 = feature_map(features)
        result2 = feature_map.circuit(features)
        
        assert result1 == result2
    
    def test_compute_kernel_matrix_method(self):
        """Test kernel matrix computation method."""
        feature_map = QuantumKernelFeatureMap("zz", 2, 2, seed=42)
        
        X = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        kernel = feature_map.compute_kernel_matrix(X)
        
        assert kernel.shape == (2, 2)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
    
    def test_get_feature_vector_method(self):
        """Test feature vector extraction."""
        feature_map = QuantumKernelFeatureMap("iqp", 3, 2, seed=42)
        
        sample = torch.tensor([0.5, 1.0])
        feature_vector = feature_map.get_feature_vector(sample)
        
        assert isinstance(feature_vector, torch.Tensor)
        assert len(feature_vector) == 3  # n_qubits
    
    def test_get_info_method(self):
        """Test information retrieval."""
        feature_map = QuantumKernelFeatureMap("zz", 4, 3, repetitions=2, 
                                             entanglement="circular")
        
        info = feature_map.get_info()
        
        assert info["feature_map_type"] == "zz"
        assert info["n_qubits"] == 4
        assert info["n_features"] == 3
        assert info["repetitions"] == 2
        assert info["entanglement"] == "circular"
        assert "complexity_metrics" in info
    
    def test_get_complexity_summary(self):
        """Test complexity summary generation."""
        feature_map = QuantumKernelFeatureMap("iqp", 3, 3)
        
        summary = feature_map.get_complexity_summary()
        
        assert isinstance(summary, str)
        assert "IQP" in summary
        assert "Qubits: 3" in summary
        assert "Total Gates:" in summary


class TestQKSRandomParameters:
    """Test suite for QKS random parameter generation."""
    
    def test_generate_random_parameters_uniform(self):
        """Test uniform random parameter generation."""
        params = generate_random_parameters(4, 3, 5, "uniform", seed=42)
        
        expected_keys = ["rotation_params", "data_params", "entanglement_params"]
        for key in expected_keys:
            assert key in params
            assert isinstance(params[key], torch.Tensor)
        
        # Check shapes
        assert params["rotation_params"].shape == (3, 4, 3)  # (n_layers, n_qubits, 3)
        assert params["data_params"].shape == (5, 4)  # (n_features, n_qubits)
        assert params["entanglement_params"].shape == (3, 4)  # (n_layers, n_qubits)
        
        # Check ranges for uniform distribution [0, 2π]
        for key in expected_keys:
            assert torch.all(params[key] >= 0)
            assert torch.all(params[key] <= 2 * np.pi)
    
    def test_generate_random_parameters_normal(self):
        """Test normal random parameter generation."""
        params = generate_random_parameters(3, 2, 4, "normal", seed=42)
        
        # Normal distribution should have both positive and negative values
        rotation_params = params["rotation_params"]
        assert torch.any(rotation_params > 0)
        assert torch.any(rotation_params < 0)
    
    def test_random_parameters_reproducibility(self):
        """Test reproducibility with same seed."""
        params1 = generate_random_parameters(3, 2, 4, "uniform", seed=42)
        params2 = generate_random_parameters(3, 2, 4, "uniform", seed=42)
        
        for key in params1:
            assert torch.allclose(params1[key], params2[key])


class TestQKSCircuit:
    """Test suite for QKS circuit construction."""
    
    def test_build_qks_circuit_basic(self):
        """Test basic QKS circuit construction."""
        n_qubits, n_layers, n_features = 3, 2, 4
        circuit, params = build_qks_circuit(n_qubits, n_layers, n_features, seed=42)
        
        assert callable(circuit)
        assert isinstance(params, dict)
        
        # Test with features
        features = torch.randn(n_features)
        result = circuit(features)
        
        assert isinstance(result, list)
        assert len(result) == n_qubits
        
        for exp_val in result:
            assert -1 <= exp_val <= 1
    
    def test_qks_circuit_different_entanglements(self):
        """Test QKS circuit with different entanglement patterns."""
        patterns = ["linear", "random", "all_to_all"]
        features = torch.randn(3)
        
        for pattern in patterns:
            circuit, _ = build_qks_circuit(3, 2, 3, entanglement_pattern=pattern, seed=42)
            result = circuit(features)
            
            assert len(result) == 3
            for exp_val in result:
                assert -1 <= exp_val <= 1
    
    def test_qks_feature_padding_truncation(self):
        """Test QKS with different feature lengths."""
        circuit, _ = build_qks_circuit(3, 2, 4, seed=42)
        
        # Test with fewer features (should pad)
        short_features = torch.tensor([1.0, 2.0])
        result_short = circuit(short_features)
        assert len(result_short) == 3
        
        # Test with more features (should truncate)
        long_features = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result_long = circuit(long_features)
        assert len(result_long) == 3


class TestQKSFeatureComputation:
    """Test suite for QKS feature computation."""
    
    def test_compute_qks_features(self):
        """Test QKS features computation for dataset."""
        n_qubits, n_layers, n_features = 3, 2, 3
        circuit, _ = build_qks_circuit(n_qubits, n_layers, n_features, seed=42)
        
        X = torch.randn(5, n_features)  # 5 samples
        features = compute_qks_features(circuit, X)
        
        assert features.shape == (5, n_qubits)
        
        # All features should be in reasonable range
        assert torch.all(features >= -1.1)  # Allow small numerical errors
        assert torch.all(features <= 1.1)
    
    def test_compute_qks_kernel_approximation(self):
        """Test QKS kernel approximation."""
        circuit, _ = build_qks_circuit(3, 2, 3, seed=42)
        
        X = torch.randn(4, 3)
        kernel = compute_qks_kernel_approximation(circuit, X)
        
        assert kernel.shape == (4, 4)
        
        # Should be symmetric
        assert torch.allclose(kernel, kernel.T, atol=1e-6)


class TestQKSComplexity:
    """Test suite for QKS complexity benchmarking."""
    
    def test_benchmark_qks_complexity(self):
        """Test QKS complexity benchmarking."""
        complexity = benchmark_qks_complexity(4, 3, 5, "linear")
        
        expected_keys = [
            "n_qubits", "n_layers", "n_features", "total_gates",
            "rotation_gates", "cnot_gates", "circuit_depth"
        ]
        
        for key in expected_keys:
            assert key in complexity
            assert isinstance(complexity[key], int)
            assert complexity[key] >= 0
        
        assert complexity["n_qubits"] == 4
        assert complexity["n_layers"] == 3
        assert complexity["total_gates"] > 0
    
    def test_complexity_scaling_with_entanglement(self):
        """Test complexity scaling with different entanglement patterns."""
        patterns = ["linear", "random", "all_to_all"]
        complexities = []
        
        for pattern in patterns:
            complexity = benchmark_qks_complexity(4, 2, 4, pattern)
            complexities.append(complexity["total_gates"])
        
        # All-to-all should generally be most complex
        assert complexities[2] >= complexities[0]  # all_to_all >= linear


class TestQuantumKitchenSinks:
    """Test suite for QuantumKitchenSinks wrapper class."""
    
    def test_qks_initialization(self):
        """Test QKS initialization."""
        qks = QuantumKitchenSinks(4, 3, 5, "linear", "uniform", seed=42)
        
        assert qks.n_qubits == 4
        assert qks.n_layers == 3
        assert qks.n_features == 5
        assert callable(qks.circuit)
        assert isinstance(qks.complexity, dict)
    
    def test_qks_callable_interface(self):
        """Test QKS callable interface."""
        qks = QuantumKitchenSinks(3, 2, 3, seed=42)
        features = torch.randn(3)
        
        result1 = qks(features)
        result2 = qks.circuit(features)
        
        assert result1 == result2
    
    def test_qks_compute_features(self):
        """Test QKS features computation method."""
        qks = QuantumKitchenSinks(3, 2, 4, seed=42)
        
        X = torch.randn(5, 4)
        features = qks.compute_features(X)
        
        assert features.shape == (5, 3)
    
    def test_qks_compute_kernel_approximation(self):
        """Test QKS kernel approximation method."""
        qks = QuantumKitchenSinks(3, 2, 3, seed=42)
        
        X = torch.randn(4, 3)
        kernel = qks.compute_kernel_approximation(X)
        
        assert kernel.shape == (4, 4)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
    
    def test_qks_get_info(self):
        """Test QKS information retrieval."""
        qks = QuantumKitchenSinks(4, 3, 5, "random", "normal", seed=42)
        
        info = qks.get_info()
        
        assert info["n_qubits"] == 4
        assert info["n_layers"] == 3
        assert info["entanglement_pattern"] == "random"
        assert info["parameter_type"] == "normal"
        assert "complexity_metrics" in info
    
    def test_qks_get_complexity_summary(self):
        """Test QKS complexity summary."""
        qks = QuantumKitchenSinks(3, 2, 4, seed=42)
        
        summary = qks.get_complexity_summary()
        
        assert isinstance(summary, str)
        assert "Quantum Kitchen Sinks Complexity:" in summary
        assert "Qubits: 3" in summary


class TestEnsembleQKS:
    """Test suite for Ensemble QKS."""
    
    def test_ensemble_initialization(self):
        """Test ensemble QKS initialization."""
        ensemble = EnsembleQKS(3, 2, 4, n_estimators=5, base_seed=42)
        
        assert len(ensemble.qks_ensemble) == 5
        assert ensemble.n_estimators == 5
        assert ensemble.n_qubits == 3
    
    def test_ensemble_features(self):
        """Test ensemble features computation."""
        ensemble = EnsembleQKS(3, 2, 4, n_estimators=3, base_seed=42)
        
        X = torch.randn(4, 4)
        features = ensemble.compute_ensemble_features(X)
        
        # Should concatenate features from all estimators
        assert features.shape == (4, 3 * 3)  # (n_samples, n_estimators * n_qubits)
    
    def test_ensemble_kernel(self):
        """Test ensemble kernel computation."""
        ensemble = EnsembleQKS(2, 2, 3, n_estimators=3, base_seed=42)
        
        X = torch.randn(3, 3)
        kernel = ensemble.compute_ensemble_kernel(X)
        
        assert kernel.shape == (3, 3)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
    
    def test_ensemble_info(self):
        """Test ensemble information retrieval."""
        ensemble = EnsembleQKS(3, 2, 4, n_estimators=5, base_seed=42)
        
        info = ensemble.get_info()
        
        assert info["n_estimators"] == 5
        assert info["n_qubits_per_estimator"] == 3
        assert info["total_qubits"] == 15  # 5 * 3


class TestPCA16Integration:
    """Integration tests with PCA-16 dataset."""
    
    def test_zz_feature_map_pca16(self):
        """Test ZZ feature map with PCA-16 dataset."""
        # Create mock PCA-16 dataset
        pca_16_data = torch.randn(10, 16)  # 10 samples, 16 features
        
        # Use 4 qubits to handle 4 features (subset of PCA-16)
        feature_map = QuantumKernelFeatureMap("zz", n_qubits=4, n_features=4, 
                                             repetitions=2, seed=42)
        
        # Use first 4 features of PCA-16 data
        X_subset = pca_16_data[:, :4]
        
        kernel = feature_map.compute_kernel_matrix(X_subset)
        
        assert kernel.shape == (10, 10)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
        
        # Diagonal should represent self-similarity
        diagonal = torch.diag(kernel)
        assert torch.all(diagonal > 0.3)  # Reasonable self-similarity
    
    def test_iqp_feature_map_pca16(self):
        """Test IQP feature map with PCA-16 dataset."""
        pca_16_data = torch.randn(8, 16)
        
        # Use 4 qubits for 4 features
        feature_map = QuantumKernelFeatureMap("iqp", n_qubits=4, n_features=4, seed=42)
        
        X_subset = pca_16_data[:, :4]
        
        # Test feature vector extraction
        for i in range(len(X_subset)):
            feature_vector = feature_map.get_feature_vector(X_subset[i])
            assert len(feature_vector) == 4
            assert torch.all(torch.abs(feature_vector) <= 1.1)
    
    def test_qks_pca16(self):
        """Test QKS with PCA-16 dataset."""
        pca_16_data = torch.randn(12, 16)
        
        # QKS can handle full 16 features with fewer qubits
        qks = QuantumKitchenSinks(n_qubits=5, n_layers=3, n_features=16, 
                                 entanglement_pattern="linear", seed=42)
        
        # Compute QKS features
        qks_features = qks.compute_features(pca_16_data)
        
        assert qks_features.shape == (12, 5)  # 12 samples, 5 qubits
        
        # Compute kernel approximation
        kernel = qks.compute_kernel_approximation(pca_16_data)
        
        assert kernel.shape == (12, 12)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
    
    def test_ensemble_qks_pca16(self):
        """Test ensemble QKS with PCA-16 dataset."""
        pca_16_data = torch.randn(6, 16)
        
        ensemble = EnsembleQKS(n_qubits=4, n_layers=2, n_features=16, 
                              n_estimators=5, base_seed=42)
        
        # Compute ensemble features
        ensemble_features = ensemble.compute_ensemble_features(pca_16_data)
        
        assert ensemble_features.shape == (6, 20)  # 6 samples, 5*4 features
        
        # Compute ensemble kernel
        kernel = ensemble.compute_ensemble_kernel(pca_16_data)
        
        assert kernel.shape == (6, 6)
        assert torch.allclose(kernel, kernel.T, atol=1e-6)
    
    def test_feature_map_comparison_pca16(self):
        """Compare different feature maps on PCA-16 subset."""
        pca_16_subset = torch.randn(5, 4)  # 5 samples, 4 features
        
        # Different feature maps
        zz_map = QuantumKernelFeatureMap("zz", 4, 4, seed=42)
        iqp_map = QuantumKernelFeatureMap("iqp", 4, 4, seed=42)
        qks = QuantumKitchenSinks(4, 2, 4, seed=42)
        
        # Compute kernels
        zz_kernel = zz_map.compute_kernel_matrix(pca_16_subset)
        iqp_kernel = iqp_map.compute_kernel_matrix(pca_16_subset)
        qks_kernel = qks.compute_kernel_approximation(pca_16_subset)
        
        # All should be valid kernel matrices
        for kernel in [zz_kernel, iqp_kernel, qks_kernel]:
            assert kernel.shape == (5, 5)
            assert torch.allclose(kernel, kernel.T, atol=1e-6)
        
        # Kernels should be different (different encoding methods)
        assert not torch.allclose(zz_kernel, iqp_kernel, atol=1e-1)
        assert not torch.allclose(zz_kernel, qks_kernel, atol=1e-1)
    
    def test_complexity_analysis_pca16(self):
        """Test complexity analysis for PCA-16 configurations."""
        configurations = [
            ("zz", 4, 4, 2, "linear"),
            ("iqp", 4, 4, 1, None),
            ("zz", 5, 5, 1, "circular"),
            ("iqp", 3, 3, 2, None),
        ]
        
        for config in configurations:
            if config[0] == "zz":
                feature_map = QuantumKernelFeatureMap(config[0], config[1], config[2], 
                                                     repetitions=config[3], 
                                                     entanglement=config[4])
            else:
                feature_map = QuantumKernelFeatureMap(config[0], config[1], config[2], 
                                                     repetitions=config[3])
            
            complexity = feature_map.complexity
            
            # Basic sanity checks
            assert complexity["total_gates"] > 0
            assert complexity["circuit_depth"] > 0
            assert complexity["n_qubits"] == config[1]
            assert complexity["n_features"] == config[2]
        
        # QKS complexity
        qks = QuantumKitchenSinks(4, 3, 16, "linear", seed=42)
        qks_complexity = qks.complexity
        
        assert qks_complexity["total_gates"] > 0
        assert qks_complexity["n_qubits"] == 4
        assert qks_complexity["n_features"] == 16


if __name__ == "__main__":
    pytest.main([__file__])