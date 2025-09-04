"""
Unit tests for hybrid encoding implementations.

This module tests the functionality of hybrid encoding methods that combine
angle encoding (AE) and amplitude encoding (AMP) for richer expressivity.
"""

import pytest
import torch
import numpy as np
import pennylane as qml
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_encodings.hybrid_encoding import (
    split_features_for_hybrid,
    validate_hybrid_encoding_params,
    build_hybrid_classifier,
    apply_cross_entanglement,
    apply_hybrid_variational_layer,
    apply_hybrid_entanglement,
    get_hybrid_weights_shape,
    initialize_hybrid_weights,
    benchmark_hybrid_circuit,
    create_hybrid_model,
    HybridEncodingClassifier
)


class TestUtilityFunctions:
    """Test suite for hybrid encoding utility functions."""
    
    def test_split_features_for_hybrid_exact_split(self):
        """Test feature splitting with exact dimensions."""
        features = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        angle_features, amp_features = split_features_for_hybrid(features, 2, 4)
        
        expected_angle = torch.tensor([1.0, 2.0])
        expected_amp = torch.tensor([3.0, 4.0, 5.0, 6.0])
        
        assert torch.allclose(angle_features, expected_angle)
        assert torch.allclose(amp_features, expected_amp)
    
    def test_split_features_for_hybrid_padding(self):
        """Test feature splitting with padding."""
        features = torch.tensor([1.0, 2.0, 3.0])
        angle_features, amp_features = split_features_for_hybrid(features, 2, 4)
        
        expected_angle = torch.tensor([1.0, 2.0])
        expected_amp = torch.tensor([3.0, 0.0, 0.0, 0.0])
        
        assert torch.allclose(angle_features, expected_angle)
        assert torch.allclose(amp_features, expected_amp)
    
    def test_split_features_for_hybrid_truncation(self):
        """Test feature splitting with truncation."""
        features = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        angle_features, amp_features = split_features_for_hybrid(features, 3, 2)
        
        expected_angle = torch.tensor([1.0, 2.0, 3.0])
        expected_amp = torch.tensor([4.0, 5.0])
        
        assert torch.allclose(angle_features, expected_angle)
        assert torch.allclose(amp_features, expected_amp)
    
    def test_split_features_numpy(self):
        """Test feature splitting with numpy arrays."""
        features = np.array([1.0, 2.0, 3.0, 4.0])
        angle_features, amp_features = split_features_for_hybrid(features, 2, 2)
        
        expected_angle = np.array([1.0, 2.0])
        expected_amp = np.array([3.0, 4.0])
        
        assert np.allclose(angle_features, expected_angle)
        assert np.allclose(amp_features, expected_amp)
    
    def test_validate_hybrid_encoding_params_valid(self):
        """Test validation with valid parameters."""
        # Should not raise any exception
        validate_hybrid_encoding_params(3, 2)
        validate_hybrid_encoding_params(4, 4)
        validate_hybrid_encoding_params(1, 1)
    
    def test_validate_hybrid_encoding_params_invalid(self):
        """Test validation with invalid parameters."""
        with pytest.raises(ValueError, match="n_angle_qubits must be positive"):
            validate_hybrid_encoding_params(0, 2)
        
        with pytest.raises(ValueError, match="n_amplitude_qubits must be positive"):
            validate_hybrid_encoding_params(3, 0)
        
        with pytest.raises(ValueError, match="Total qubits should not exceed 20"):
            validate_hybrid_encoding_params(15, 10)


class TestHybridCircuitConstruction:
    """Test suite for hybrid circuit construction."""
    
    def test_build_hybrid_classifier_basic(self):
        """Test basic hybrid classifier construction."""
        n_angle_qubits, n_amplitude_qubits = 2, 2
        circuit = build_hybrid_classifier(n_angle_qubits, n_amplitude_qubits)
        
        assert callable(circuit)
        
        # Test with appropriate inputs
        # 2 angle features + 4 amplitude features (2^2)
        features = torch.randn(6)
        weights = torch.randn(2, 4, 3)  # 2 layers, 4 total qubits, 3 rotations each
        
        result = circuit(features, weights)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        assert -1 <= result <= 1
    
    def test_build_hybrid_classifier_different_strategies(self):
        """Test hybrid classifier with different entanglement strategies."""
        n_angle_qubits, n_amplitude_qubits = 2, 2
        strategies = ["linear", "circular", "full"]
        
        for strategy in strategies:
            circuit = build_hybrid_classifier(n_angle_qubits, n_amplitude_qubits, 
                                             entanglement_strategy=strategy)
            
            features = torch.randn(6)  # 2 + 4 features
            weights = torch.randn(2, 4, 3)
            
            result = circuit(features, weights)
            
            assert isinstance(result, torch.Tensor)
            assert -1 <= result <= 1
    
    def test_build_hybrid_classifier_invalid_inputs(self):
        """Test hybrid classifier with invalid inputs."""
        with pytest.raises(ValueError):
            build_hybrid_classifier(0, 2)
        
        with pytest.raises(ValueError):
            build_hybrid_classifier(2, 0)


class TestHybridComponents:
    """Test suite for hybrid encoding components."""
    
    def test_apply_cross_entanglement_linear(self):
        """Test linear cross-entanglement."""
        device = qml.device("default.qubit", wires=4)
        
        @qml.qnode(device)
        def test_circuit():
            # Initialize with Hadamards
            for i in range(4):
                qml.Hadamard(wires=i)
            apply_cross_entanglement(2, 2, "linear")
            return qml.state()
        
        state = test_circuit()
        
        # State should be normalized
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        
        norm_squared = torch.sum(torch.abs(state)**2)
        assert torch.allclose(norm_squared, torch.tensor(1.0, dtype=norm_squared.dtype), atol=1e-6)
    
    def test_apply_hybrid_variational_layer(self):
        """Test hybrid variational layer."""
        n_qubits = 3
        device = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(device)
        def test_circuit(weights):
            apply_hybrid_variational_layer(weights, n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weights = torch.tensor([
            [np.pi/4, np.pi/2, 0.0],
            [0.0, 0.0, np.pi/2],
            [np.pi, np.pi/2, np.pi]
        ])
        
        expectations = test_circuit(weights)
        
        # Check that expectations are within valid range
        for exp in expectations:
            assert -1 <= exp <= 1
    
    def test_apply_hybrid_entanglement_strategies(self):
        """Test different entanglement strategies."""
        n_qubits = 4
        device = qml.device("default.qubit", wires=n_qubits)
        
        strategies = ["linear", "circular", "full"]
        
        for strategy in strategies:
            @qml.qnode(device)
            def test_circuit():
                # Initialize with Hadamards
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                apply_hybrid_entanglement(n_qubits, strategy)
                return qml.state()
            
            state = test_circuit()
            
            # State should be normalized
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            
            norm_squared = torch.sum(torch.abs(state)**2)
            assert torch.allclose(norm_squared, torch.tensor(1.0, dtype=norm_squared.dtype), atol=1e-6)


class TestWeightManagement:
    """Test suite for weight management."""
    
    def test_get_hybrid_weights_shape(self):
        """Test weight shape calculation."""
        shape = get_hybrid_weights_shape(n_qubits=5, n_layers=3)
        expected_shape = (3, 5, 3)
        assert shape == expected_shape
    
    def test_initialize_hybrid_weights(self):
        """Test weight initialization."""
        n_qubits, n_layers = 4, 2
        weights = initialize_hybrid_weights(n_qubits, n_layers, seed=42)
        
        # Check shape
        expected_shape = (n_layers, n_qubits, 3)
        assert weights.shape == expected_shape
        
        # Check gradients are enabled
        assert weights.requires_grad is True
        
        # Check reproducibility
        weights2 = initialize_hybrid_weights(n_qubits, n_layers, seed=42)
        assert torch.allclose(weights, weights2)
        
        # Check different seed gives different weights
        weights3 = initialize_hybrid_weights(n_qubits, n_layers, seed=123)
        assert not torch.allclose(weights, weights3)


class TestComplexityBenchmarking:
    """Test suite for complexity benchmarking."""
    
    def test_benchmark_hybrid_circuit_basic(self):
        """Test basic complexity benchmarking."""
        metrics = benchmark_hybrid_circuit(n_angle_qubits=2, n_amplitude_qubits=2, 
                                         n_layers=2, entanglement_strategy="linear")
        
        # Check that all expected keys are present
        expected_keys = [
            "total_qubits", "total_gates", "circuit_depth",
            "angle_encoding_gates", "amplitude_encoding_gates",
            "cross_entanglement_gates", "variational_gates", "entanglement_gates"
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], int)
            assert metrics[key] >= 0
        
        # Check basic relationships
        assert metrics["total_qubits"] == 4
        assert metrics["angle_encoding_gates"] == 2
        assert metrics["total_gates"] > 0
        assert metrics["circuit_depth"] > 0
    
    def test_benchmark_different_strategies(self):
        """Test benchmarking with different entanglement strategies."""
        strategies = ["linear", "circular", "full"]
        
        for strategy in strategies:
            metrics = benchmark_hybrid_circuit(2, 2, 2, strategy)
            
            assert metrics["total_qubits"] == 4
            assert metrics["total_gates"] > 0
            
            # Full strategy should have more gates
            if strategy == "full":
                linear_metrics = benchmark_hybrid_circuit(2, 2, 2, "linear")
                assert metrics["total_gates"] >= linear_metrics["total_gates"]


class TestModelCreation:
    """Test suite for complete model creation."""
    
    def test_create_hybrid_model_basic(self):
        """Test basic hybrid model creation."""
        circuit, weights, complexity = create_hybrid_model(
            n_angle_features=3, n_amplitude_features_log=2, 
            n_layers=2, seed=42
        )
        
        assert callable(circuit)
        
        # Check weights shape (3 angle + 2 amplitude qubits = 5 total)
        expected_shape = (2, 5, 3)
        assert weights.shape == expected_shape
        assert weights.requires_grad is True
        
        # Check complexity metrics
        assert isinstance(complexity, dict)
        assert complexity["total_qubits"] == 5
        
        # Test forward pass
        features = torch.randn(7)  # 3 angle + 4 amplitude features
        result = circuit(features, weights)
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1
    
    def test_create_hybrid_model_different_configs(self):
        """Test model creation with different configurations."""
        configs = [
            (2, 1, 1),  # Small model
            (4, 2, 3),  # Medium model
            (3, 3, 2),  # Balanced model
        ]
        
        for n_angle, n_amp_log, n_layers in configs:
            circuit, weights, complexity = create_hybrid_model(
                n_angle, n_amp_log, n_layers, seed=42
            )
            
            total_qubits = n_angle + n_amp_log
            expected_shape = (n_layers, total_qubits, 3)
            
            assert weights.shape == expected_shape
            assert complexity["total_qubits"] == total_qubits


class TestHybridEncodingClassifier:
    """Test suite for HybridEncodingClassifier wrapper class."""
    
    def test_initialization(self):
        """Test hybrid classifier initialization."""
        classifier = HybridEncodingClassifier(
            n_angle_features=3, n_amplitude_features_log=2,
            n_layers=2, seed=42
        )
        
        assert classifier.n_angle_features == 3
        assert classifier.n_amplitude_features_log == 2
        assert classifier.n_amplitude_features == 4
        assert classifier.n_total_features == 7
        assert classifier.n_qubits == 5
        assert classifier.n_layers == 2
        assert callable(classifier.circuit)
        assert classifier.weights.requires_grad is True
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        classifier = HybridEncodingClassifier(
            n_angle_features=2, n_amplitude_features_log=2,
            n_layers=1, seed=42
        )
        
        # Single sample (2 angle + 4 amplitude = 6 features)
        features = torch.randn(6)
        prediction = classifier.forward(features)
        
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == torch.Size([])
        assert -1 <= prediction <= 1
    
    def test_forward_batch(self):
        """Test forward pass with batch of samples."""
        classifier = HybridEncodingClassifier(
            n_angle_features=2, n_amplitude_features_log=1,
            n_layers=1, seed=42
        )
        
        # Batch of samples (2 angle + 2 amplitude = 4 features)
        batch_size = 3
        features = torch.randn(batch_size, 4)
        predictions = classifier.forward(features)
        
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == torch.Size([batch_size])
        assert torch.all((predictions >= -1) & (predictions <= 1))
    
    def test_callable_interface(self):
        """Test callable interface."""
        classifier = HybridEncodingClassifier(
            n_angle_features=2, n_amplitude_features_log=1, n_layers=1
        )
        
        features = torch.randn(4)
        result1 = classifier(features)
        result2 = classifier.forward(features)
        
        assert torch.allclose(result1, result2)
    
    def test_get_set_params(self):
        """Test parameter getter and setter."""
        classifier = HybridEncodingClassifier(
            n_angle_features=2, n_amplitude_features_log=1,
            n_layers=1, seed=42
        )
        
        # Get parameters
        original_params = classifier.get_params()
        assert not original_params.requires_grad
        
        # Modify parameters
        new_params = original_params * 2
        classifier.set_params(new_params)
        
        # Check update
        current_params = classifier.weights
        assert torch.allclose(current_params, new_params)
        assert current_params.requires_grad is True
    
    def test_get_circuit_info(self):
        """Test circuit information retrieval."""
        classifier = HybridEncodingClassifier(
            n_angle_features=3, n_amplitude_features_log=2,
            n_layers=2, entanglement_strategy="linear"
        )
        
        info = classifier.get_circuit_info()
        
        # Check basic info
        assert info["n_angle_features"] == 3
        assert info["n_amplitude_features"] == 4
        assert info["n_total_features"] == 7
        assert info["n_qubits"] == 5
        assert info["n_layers"] == 2
        assert info["encoding_type"] == "Hybrid (Angle + Amplitude)"
        assert info["entanglement_strategy"] == "linear"
        
        # Check complexity metrics are included
        assert "complexity_metrics" in info
        assert isinstance(info["complexity_metrics"], dict)
    
    def test_get_complexity_summary(self):
        """Test complexity summary generation."""
        classifier = HybridEncodingClassifier(
            n_angle_features=2, n_amplitude_features_log=2, n_layers=1
        )
        
        summary = classifier.get_complexity_summary()
        
        assert isinstance(summary, str)
        assert "Total Qubits:" in summary
        assert "Total Gates:" in summary
        assert "Circuit Depth:" in summary
        assert "Gate Breakdown:" in summary


class TestPCA16Integration:
    """Integration tests with PCA-16 subset."""
    
    def test_pca_16_hybrid_classification(self):
        """Test classification with 16-dimensional PCA vector (hybrid)."""
        # Create PCA-16 feature vector
        pca_16 = torch.randn(16)
        
        # Configure hybrid: 8 angle features + 3 amplitude qubits (2^3=8 features)
        classifier = HybridEncodingClassifier(
            n_angle_features=8, n_amplitude_features_log=3,
            n_layers=2, seed=42
        )
        
        prediction = classifier(pca_16)
        
        assert isinstance(prediction, torch.Tensor)
        assert -1 <= prediction <= 1
        
        # Test with batch
        batch_pca = torch.randn(5, 16)
        batch_predictions = classifier(batch_pca)
        
        assert batch_predictions.shape == torch.Size([5])
        assert torch.all((batch_predictions >= -1) & (batch_predictions <= 1))
    
    def test_pca_16_different_splits(self):
        """Test different ways to split PCA-16 features."""
        pca_16 = torch.randn(16)
        
        # Different hybrid configurations for 16 features
        configs = [
            (12, 2, 2),  # 12 angle + 4 amplitude features
            (8, 3, 2),   # 8 angle + 8 amplitude features
            (4, 2, 1),   # 4 angle + 4 amplitude features (truncated)
        ]
        
        for n_angle, n_amp_log, n_layers in configs:
            classifier = HybridEncodingClassifier(
                n_angle_features=n_angle, n_amplitude_features_log=n_amp_log,
                n_layers=n_layers, seed=42
            )
            
            prediction = classifier(pca_16)
            
            assert isinstance(prediction, torch.Tensor)
            assert -1 <= prediction <= 1
    
    def test_gradient_computation_pca_16(self):
        """Test gradient computation with PCA-16 vectors."""
        classifier = HybridEncodingClassifier(
            n_angle_features=8, n_amplitude_features_log=3,
            n_layers=2, seed=42
        )
        
        pca_features = torch.randn(16, requires_grad=True)
        
        prediction = classifier(pca_features)
        prediction.backward()
        
        # Check gradients exist
        assert pca_features.grad is not None
        assert classifier.weights.grad is not None
    
    def test_training_loop_simulation_pca_16(self):
        """Test training loop compatibility with PCA-16 vectors."""
        classifier = HybridEncodingClassifier(
            n_angle_features=8, n_amplitude_features_log=3,
            n_layers=2, seed=42
        )
        
        # Create dummy PCA-16 data
        X = torch.randn(6, 16)  # 6 samples, 16 PCA features
        y = torch.randint(0, 2, (6,)).float() * 2 - 1  # Binary labels {-1, +1}
        
        # Define loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([classifier.weights], lr=0.1)
        
        # Training step
        initial_loss = float('inf')
        for epoch in range(3):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = classifier(X)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        # Check that loss decreased
        final_loss = loss.item()
        assert final_loss < initial_loss, "Model should be learning"
    
    def test_complexity_analysis_pca_16(self):
        """Test complexity analysis for PCA-16 hybrid models."""
        # Different configurations for PCA-16
        configs = [
            (8, 3, 1, "linear"),
            (8, 3, 2, "linear"),
            (8, 3, 1, "circular"),
            (8, 3, 1, "full"),
        ]
        
        for n_angle, n_amp_log, n_layers, strategy in configs:
            classifier = HybridEncodingClassifier(
                n_angle_features=n_angle, n_amplitude_features_log=n_amp_log,
                n_layers=n_layers, entanglement_strategy=strategy, seed=42
            )
            
            info = classifier.get_circuit_info()
            complexity = info["complexity_metrics"]
            
            # Basic sanity checks
            assert complexity["total_qubits"] == n_angle + n_amp_log
            assert complexity["total_gates"] > 0
            assert complexity["circuit_depth"] > 0
            
            # More complex strategies should generally have more gates
            if strategy == "full":
                assert complexity["total_gates"] > 20  # Should be reasonably complex


if __name__ == "__main__":
    pytest.main([__file__])