"""
Unit tests for amplitude encoding implementations.

This module tests the functionality of amplitude encoding methods,
including exact and approximate implementations with PCA vectors.
"""

import pytest
import torch
import numpy as np
import pennylane as qml
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_encodings.amplitude_encoding import (
    normalize_vector,
    pad_or_truncate_vector,
    validate_amplitude_encoding_size,
    build_exact_amplitude_classifier,
    build_approximate_amplitude_classifier,
    get_amplitude_weights_shape,
    initialize_amplitude_weights,
    create_amplitude_model,
    AmplitudeEncodingClassifier
)


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_normalize_vector_l2_torch(self):
        """Test L2 normalization with torch tensors."""
        vector = torch.tensor([3.0, 4.0])
        normalized = normalize_vector(vector, method="l2")
        
        # Check L2 norm is 1
        norm = torch.norm(normalized)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
        
        # Check direction is preserved
        expected = torch.tensor([0.6, 0.8])
        assert torch.allclose(normalized, expected, atol=1e-6)
    
    def test_normalize_vector_l2_numpy(self):
        """Test L2 normalization with numpy arrays."""
        vector = np.array([3.0, 4.0])
        normalized = normalize_vector(vector, method="l2")
        
        # Check L2 norm is 1
        norm = np.linalg.norm(normalized)
        assert np.allclose(norm, 1.0, atol=1e-6)
        
        # Check direction is preserved
        expected = np.array([0.6, 0.8])
        assert np.allclose(normalized, expected, atol=1e-6)
    
    def test_normalize_vector_unit_sum(self):
        """Test unit sum normalization."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        normalized = normalize_vector(vector, method="unit")
        
        # Check sum of absolute values is 1
        sum_abs = torch.sum(torch.abs(normalized))
        assert torch.allclose(sum_abs, torch.tensor(1.0), atol=1e-6)
        
        expected = torch.tensor([1/6, 2/6, 3/6])
        assert torch.allclose(normalized, expected, atol=1e-6)
    
    def test_normalize_vector_zero_vector(self):
        """Test normalization with zero vector."""
        vector = torch.zeros(3)
        normalized = normalize_vector(vector, method="l2")
        
        # Should return zero vector unchanged
        assert torch.allclose(normalized, vector)
    
    def test_normalize_vector_invalid_method(self):
        """Test invalid normalization method."""
        vector = torch.tensor([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_vector(vector, method="invalid")
    
    def test_pad_or_truncate_vector_padding(self):
        """Test vector padding with zeros."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        result = pad_or_truncate_vector(vector, target_size=5)
        
        expected = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_pad_or_truncate_vector_truncating(self):
        """Test vector truncation."""
        vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_or_truncate_vector(vector, target_size=3)
        
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected)
    
    def test_pad_or_truncate_vector_same_size(self):
        """Test vector with same size as target."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        result = pad_or_truncate_vector(vector, target_size=3)
        
        assert torch.allclose(result, vector)
    
    def test_validate_amplitude_encoding_size(self):
        """Test qubit count validation for amplitude encoding."""
        # Test various input sizes
        test_cases = [
            (1, 0),    # 2^0 = 1 amplitude
            (2, 1),    # 2^1 = 2 amplitudes
            (3, 2),    # 2^2 = 4 amplitudes  
            (4, 2),    # 2^2 = 4 amplitudes
            (5, 3),    # 2^3 = 8 amplitudes
            (8, 3),    # 2^3 = 8 amplitudes
            (9, 4),    # 2^4 = 16 amplitudes
            (16, 4),   # 2^4 = 16 amplitudes
        ]
        
        for n_features, expected_qubits in test_cases:
            result = validate_amplitude_encoding_size(n_features)
            assert result == expected_qubits
    
    def test_validate_amplitude_encoding_size_invalid(self):
        """Test validation with invalid input."""
        with pytest.raises(ValueError, match="n_features must be positive"):
            validate_amplitude_encoding_size(0)
        
        with pytest.raises(ValueError, match="n_features must be positive"):
            validate_amplitude_encoding_size(-1)


class TestExactAmplitudeEncoding:
    """Test suite for exact amplitude encoding."""
    
    def test_build_exact_amplitude_classifier(self):
        """Test exact amplitude classifier construction."""
        n_qubits, n_layers = 3, 2
        circuit = build_exact_amplitude_classifier(n_qubits, n_layers)
        
        assert callable(circuit)
        
        # Test with sample inputs (8 amplitudes for 3 qubits)
        features = torch.randn(8)
        features = features / torch.norm(features)  # Pre-normalize for MottonenStatePreparation
        weights = torch.randn(n_layers, n_qubits, 2)
        
        result = circuit(features, weights)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        assert -1 <= result <= 1
    
    def test_build_exact_amplitude_classifier_invalid_inputs(self):
        """Test exact classifier with invalid inputs."""
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            build_exact_amplitude_classifier(0, 2)
        
        with pytest.raises(ValueError, match="n_layers must be positive"):
            build_exact_amplitude_classifier(3, 0)
    
    def test_exact_amplitude_encoding_with_pca_32(self):
        """Test exact amplitude encoding with 32-dimensional PCA vector."""
        # Create mock 32-dimensional PCA vector
        pca_32 = torch.randn(32)
        pca_32 = pca_32 / torch.norm(pca_32)  # Normalize
        
        n_qubits = validate_amplitude_encoding_size(32)  # Should be 5 qubits (2^5=32)
        circuit = build_exact_amplitude_classifier(n_qubits, n_layers=2)
        
        weights = initialize_amplitude_weights(n_qubits, 2, seed=42)
        
        result = circuit(pca_32, weights)
        
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1
    
    def test_exact_amplitude_encoding_with_pca_64(self):
        """Test exact amplitude encoding with 64-dimensional PCA vector."""
        # Create mock 64-dimensional PCA vector
        pca_64 = torch.randn(64)
        pca_64 = pca_64 / torch.norm(pca_64)  # Normalize
        
        n_qubits = validate_amplitude_encoding_size(64)  # Should be 6 qubits (2^6=64)
        circuit = build_exact_amplitude_classifier(n_qubits, n_layers=2)
        
        weights = initialize_amplitude_weights(n_qubits, 2, seed=42)
        
        result = circuit(pca_64, weights)
        
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1


class TestApproximateAmplitudeEncoding:
    """Test suite for approximate amplitude encoding."""
    
    def test_build_approximate_amplitude_classifier(self):
        """Test approximate amplitude classifier construction."""
        n_qubits, n_layers = 4, 2
        circuit = build_approximate_amplitude_classifier(n_qubits, n_layers)
        
        assert callable(circuit)
        
        # Test with sample inputs
        features = torch.randn(10)  # Can be any size for approximate
        weights = torch.randn(n_layers, n_qubits, 2)
        
        result = circuit(features, weights)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        assert -1 <= result <= 1
    
    def test_approximate_amplitude_encoding_with_pca_32(self):
        """Test approximate amplitude encoding with 32-dimensional PCA vector."""
        pca_32 = torch.randn(32)
        
        # Use fewer qubits for approximation
        n_qubits = 5
        circuit = build_approximate_amplitude_classifier(n_qubits, n_layers=2)
        
        weights = initialize_amplitude_weights(n_qubits, 2, seed=42)
        
        result = circuit(pca_32, weights)
        
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1
    
    def test_approximate_amplitude_encoding_with_pca_64(self):
        """Test approximate amplitude encoding with 64-dimensional PCA vector."""
        pca_64 = torch.randn(64)
        
        # Use fewer qubits for approximation
        n_qubits = 6
        circuit = build_approximate_amplitude_classifier(n_qubits, n_layers=2)
        
        weights = initialize_amplitude_weights(n_qubits, 2, seed=42)
        
        result = circuit(pca_64, weights)
        
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1


class TestAmplitudeWeights:
    """Test suite for weight management."""
    
    def test_get_amplitude_weights_shape(self):
        """Test weight shape calculation."""
        shape = get_amplitude_weights_shape(n_qubits=4, n_layers=3)
        expected_shape = (3, 4, 2)
        assert shape == expected_shape
    
    def test_initialize_amplitude_weights(self):
        """Test weight initialization."""
        n_qubits, n_layers = 5, 3
        weights = initialize_amplitude_weights(n_qubits, n_layers, seed=42)
        
        # Check shape
        expected_shape = (n_layers, n_qubits, 2)
        assert weights.shape == expected_shape
        
        # Check gradients are enabled
        assert weights.requires_grad is True
        
        # Check reproducibility
        weights2 = initialize_amplitude_weights(n_qubits, n_layers, seed=42)
        assert torch.allclose(weights, weights2)


class TestModelCreation:
    """Test suite for complete model creation."""
    
    def test_create_amplitude_model_exact(self):
        """Test exact amplitude model creation."""
        n_features, n_layers = 16, 2
        circuit, weights = create_amplitude_model(n_features, n_layers, method="exact")
        
        assert callable(circuit)
        
        expected_qubits = validate_amplitude_encoding_size(n_features)  # 4 qubits
        expected_shape = (n_layers, expected_qubits, 2)
        assert weights.shape == expected_shape
        assert weights.requires_grad is True
    
    def test_create_amplitude_model_approximate(self):
        """Test approximate amplitude model creation."""
        n_features, n_layers = 32, 3
        circuit, weights = create_amplitude_model(n_features, n_layers, method="approximate")
        
        assert callable(circuit)
        
        expected_qubits = validate_amplitude_encoding_size(n_features)  # 5 qubits
        expected_shape = (n_layers, expected_qubits, 2)
        assert weights.shape == expected_shape
        assert weights.requires_grad is True
    
    def test_create_amplitude_model_invalid_method(self):
        """Test model creation with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_amplitude_model(8, 2, method="invalid")


class TestAmplitudeEncodingClassifier:
    """Test suite for AmplitudeEncodingClassifier wrapper class."""
    
    def test_initialization_exact(self):
        """Test exact classifier initialization."""
        n_features, n_layers = 8, 2
        classifier = AmplitudeEncodingClassifier(n_features, n_layers, 
                                               method="exact", seed=42)
        
        assert classifier.n_features == n_features
        assert classifier.n_qubits == validate_amplitude_encoding_size(n_features)  # 3 qubits
        assert classifier.n_layers == n_layers
        assert classifier.method == "exact"
        assert callable(classifier.circuit)
        assert classifier.weights.requires_grad is True
    
    def test_initialization_approximate(self):
        """Test approximate classifier initialization."""
        n_features, n_layers = 64, 1
        classifier = AmplitudeEncodingClassifier(n_features, n_layers,
                                               method="approximate", seed=42)
        
        assert classifier.method == "approximate"
        assert classifier.n_qubits == validate_amplitude_encoding_size(n_features)  # 6 qubits
    
    def test_forward_single_sample_exact(self):
        """Test forward pass with single sample (exact)."""
        classifier = AmplitudeEncodingClassifier(n_features=8, n_layers=2,
                                               method="exact", seed=42)
        
        features = torch.randn(8)
        prediction = classifier.forward(features)
        
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == torch.Size([])
        assert -1 <= prediction <= 1
    
    def test_forward_batch_exact(self):
        """Test forward pass with batch (exact)."""
        classifier = AmplitudeEncodingClassifier(n_features=4, n_layers=1,
                                               method="exact", seed=42)
        
        batch_size = 3
        features = torch.randn(batch_size, 4)
        predictions = classifier.forward(features)
        
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == torch.Size([batch_size])
        assert torch.all((predictions >= -1) & (predictions <= 1))
    
    def test_forward_single_sample_approximate(self):
        """Test forward pass with single sample (approximate)."""
        classifier = AmplitudeEncodingClassifier(n_features=32, n_layers=2,
                                               method="approximate", seed=42)
        
        features = torch.randn(32)
        prediction = classifier.forward(features)
        
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == torch.Size([])
        assert -1 <= prediction <= 1
    
    def test_callable_interface(self):
        """Test callable interface."""
        classifier = AmplitudeEncodingClassifier(n_features=4, n_layers=1,
                                               method="exact")
        
        features = torch.randn(4)
        result1 = classifier(features)
        result2 = classifier.forward(features)
        
        assert torch.allclose(result1, result2)
    
    def test_get_set_params(self):
        """Test parameter getter and setter."""
        classifier = AmplitudeEncodingClassifier(n_features=4, n_layers=2,
                                               method="exact", seed=42)
        
        # Get parameters
        original_params = classifier.get_params()
        assert not original_params.requires_grad  # Should be detached
        
        # Modify parameters
        new_params = original_params * 2
        classifier.set_params(new_params)
        
        # Check update
        current_params = classifier.weights
        assert torch.allclose(current_params, new_params)
        assert current_params.requires_grad is True
    
    def test_get_circuit_info(self):
        """Test circuit information retrieval."""
        n_features, n_layers = 16, 2
        classifier = AmplitudeEncodingClassifier(n_features, n_layers, method="exact")
        
        info = classifier.get_circuit_info()
        
        expected_n_qubits = validate_amplitude_encoding_size(n_features)  # 4 qubits
        expected_info = {
            "n_features": n_features,
            "n_qubits": expected_n_qubits,
            "n_layers": n_layers,
            "n_parameters": n_layers * expected_n_qubits * 2,
            "encoding_type": "Amplitude Encoding (exact)",
            "entanglement": "Linear CNOT chain",
            "variational_gates": "RY, RZ rotations",
            "readout": "PauliZ on last qubit"
        }
        
        assert info == expected_info


class TestPCAIntegration:
    """Integration tests with PCA-like vectors."""
    
    def test_pca_32_exact_classification(self):
        """Test classification with 32-dimensional PCA vector (exact)."""
        # Simulate PCA-32 feature vector
        pca_32 = torch.randn(32)
        
        classifier = AmplitudeEncodingClassifier(n_features=32, n_layers=2,
                                               method="exact", seed=42)
        
        prediction = classifier(pca_32)
        
        assert isinstance(prediction, torch.Tensor)
        assert -1 <= prediction <= 1
        
        # Test with batch
        batch_pca = torch.randn(5, 32)
        batch_predictions = classifier(batch_pca)
        
        assert batch_predictions.shape == torch.Size([5])
        assert torch.all((batch_predictions >= -1) & (batch_predictions <= 1))
    
    def test_pca_64_approximate_classification(self):
        """Test classification with 64-dimensional PCA vector (approximate)."""
        # Simulate PCA-64 feature vector
        pca_64 = torch.randn(64)
        
        classifier = AmplitudeEncodingClassifier(n_features=64, n_layers=2,
                                               method="approximate", seed=42)
        
        prediction = classifier(pca_64)
        
        assert isinstance(prediction, torch.Tensor)
        assert -1 <= prediction <= 1
    
    def test_gradient_computation_pca(self):
        """Test gradient computation with PCA vectors."""
        classifier = AmplitudeEncodingClassifier(n_features=32, n_layers=1,
                                               method="exact", seed=42)
        
        pca_features = torch.randn(32, requires_grad=True)
        
        prediction = classifier(pca_features)
        prediction.backward()
        
        # Check gradients exist
        assert pca_features.grad is not None
        assert classifier.weights.grad is not None
    
    def test_training_loop_simulation(self):
        """Test compatibility with training loop using PCA vectors."""
        # Create classifier
        classifier = AmplitudeEncodingClassifier(n_features=16, n_layers=2,
                                               method="exact", seed=42)
        
        # Create dummy PCA data
        X = torch.randn(8, 16)  # 8 samples, 16 PCA features
        y = torch.randint(0, 2, (8,)).float() * 2 - 1  # Binary labels {-1, +1}
        
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


if __name__ == "__main__":
    pytest.main([__file__])