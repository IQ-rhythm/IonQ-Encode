"""
Unit tests for quantum encoding implementations.

This module tests the functionality of various quantum encoding methods,
focusing on the Angle Encoding (AE) implementation.
"""

import pytest
import torch
import numpy as np
import pennylane as qml
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_encodings.angle_encoding import (
    build_ae_classifier,
    encode_features,
    apply_entangling_layer,
    apply_variational_layer,
    get_ae_weights_shape,
    initialize_ae_weights,
    create_ae_model,
    AngleEncodingClassifier
)


class TestAngleEncoding:
    """Test suite for Angle Encoding implementation."""
    
    def test_build_ae_classifier_basic(self):
        """Test basic functionality of build_ae_classifier."""
        n_qubits, n_layers = 4, 2
        circuit = build_ae_classifier(n_qubits, n_layers)
        
        # Test that circuit is callable
        assert callable(circuit)
        
        # Test with sample inputs
        features = torch.randn(n_qubits)
        weights = torch.randn(n_layers, n_qubits, 2)
        
        result = circuit(features, weights)
        
        # Result should be a scalar (expectation value)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])
        
        # Expectation value should be in [-1, 1]
        assert -1 <= result <= 1
    
    def test_build_ae_classifier_invalid_inputs(self):
        """Test that build_ae_classifier raises errors for invalid inputs."""
        with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
            build_ae_classifier(n_qubits=0, n_layers=2)
        
        with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
            build_ae_classifier(n_qubits=-1, n_layers=2)
        
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            build_ae_classifier(n_qubits=4, n_layers=0)
        
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            build_ae_classifier(n_qubits=4, n_layers=-1)
    
    def test_encode_features(self):
        """Test feature encoding with RY gates."""
        n_qubits = 3
        device = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(device)
        def test_circuit(features):
            encode_features(features)
            return qml.state()
        
        # Test with known angles
        features = torch.tensor([0.0, np.pi/2, np.pi])
        state = test_circuit(features)
        
        # Convert to torch tensor if needed and check normalization
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        
        # State should be normalized - convert to appropriate dtype
        norm_squared = torch.sum(torch.abs(state)**2)
        assert torch.allclose(norm_squared, torch.tensor(1.0, dtype=norm_squared.dtype), atol=1e-6)
    
    def test_apply_entangling_layer(self):
        """Test entangling layer with CNOT gates."""
        n_qubits = 4
        device = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(device)
        def test_circuit():
            # Prepare initial state |+⟩^⊗n
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            apply_entangling_layer(n_qubits)
            return qml.state()
        
        state = test_circuit()
        
        # Convert to torch tensor if needed and check normalization
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        
        # State should be normalized - convert to appropriate dtype
        norm_squared = torch.sum(torch.abs(state)**2)
        assert torch.allclose(norm_squared, torch.tensor(1.0, dtype=norm_squared.dtype), atol=1e-6)
    
    def test_apply_variational_layer(self):
        """Test variational layer with RY and RZ gates."""
        n_qubits = 3
        device = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(device)
        def test_circuit(weights):
            apply_variational_layer(weights, n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Test with known angles
        weights = torch.tensor([
            [np.pi/2, 0.0],     # RY(π/2), RZ(0) -> expect X eigenstate
            [0.0, np.pi/2],     # RY(0), RZ(π/2) -> expect Z eigenstate  
            [np.pi, np.pi]      # RY(π), RZ(π) -> expect -Z eigenstate
        ])
        
        expectations = test_circuit(weights)
        
        # Check that expectations are within valid range [-1, 1]
        for exp in expectations:
            assert -1 <= exp <= 1
    
    def test_get_ae_weights_shape(self):
        """Test weight shape calculation."""
        shape = get_ae_weights_shape(n_qubits=4, n_layers=3)
        expected_shape = (3, 4, 2)
        assert shape == expected_shape
        
        # Test with different parameters
        shape = get_ae_weights_shape(n_qubits=8, n_layers=1)
        assert shape == (1, 8, 2)
    
    def test_initialize_ae_weights(self):
        """Test weight initialization."""
        n_qubits, n_layers = 4, 2
        weights = initialize_ae_weights(n_qubits, n_layers, seed=42)
        
        # Check shape
        expected_shape = (n_layers, n_qubits, 2)
        assert weights.shape == expected_shape
        
        # Check that weights require gradients
        assert weights.requires_grad is True
        
        # Check that weights are reasonable (not too large)
        assert torch.all(torch.abs(weights) < np.pi)
        
        # Test reproducibility with seed
        weights2 = initialize_ae_weights(n_qubits, n_layers, seed=42)
        assert torch.allclose(weights, weights2)
        
        # Test different seed gives different weights
        weights3 = initialize_ae_weights(n_qubits, n_layers, seed=123)
        assert not torch.allclose(weights, weights3)
    
    def test_create_ae_model(self):
        """Test complete model creation."""
        n_features, n_layers = 4, 3
        circuit, weights = create_ae_model(n_features, n_layers)
        
        # Check circuit is callable
        assert callable(circuit)
        
        # Check weights shape
        expected_shape = (n_layers, n_features, 2)
        assert weights.shape == expected_shape
        assert weights.requires_grad is True
        
        # Test forward pass
        features = torch.randn(n_features)
        result = circuit(features, weights)
        assert isinstance(result, torch.Tensor)
        assert -1 <= result <= 1


class TestAngleEncodingClassifier:
    """Test suite for AngleEncodingClassifier wrapper class."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        n_features, n_layers = 4, 2
        classifier = AngleEncodingClassifier(n_features, n_layers, seed=42)
        
        assert classifier.n_qubits == n_features
        assert classifier.n_layers == n_layers
        assert callable(classifier.circuit)
        assert classifier.weights.requires_grad is True
        
        expected_shape = (n_layers, n_features, 2)
        assert classifier.weights.shape == expected_shape
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        classifier = AngleEncodingClassifier(n_features=4, n_layers=2, seed=42)
        
        # Single sample
        features = torch.randn(4)
        prediction = classifier.forward(features)
        
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == torch.Size([])
        assert -1 <= prediction <= 1
    
    def test_forward_batch(self):
        """Test forward pass with batch of samples."""
        classifier = AngleEncodingClassifier(n_features=4, n_layers=2, seed=42)
        
        # Batch of samples
        batch_size = 5
        features = torch.randn(batch_size, 4)
        predictions = classifier.forward(features)
        
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == torch.Size([batch_size])
        assert torch.all((predictions >= -1) & (predictions <= 1))
    
    def test_callable_interface(self):
        """Test that classifier is callable."""
        classifier = AngleEncodingClassifier(n_features=3, n_layers=1)
        
        features = torch.randn(3)
        result1 = classifier(features)
        result2 = classifier.forward(features)
        
        assert torch.allclose(result1, result2)
    
    def test_get_set_params(self):
        """Test parameter getter and setter."""
        classifier = AngleEncodingClassifier(n_features=3, n_layers=2, seed=42)
        
        # Get parameters
        original_params = classifier.get_params()
        assert not original_params.requires_grad  # Should be detached
        
        # Modify parameters
        new_params = original_params * 2
        classifier.set_params(new_params)
        
        # Check that parameters were updated
        current_params = classifier.weights
        assert torch.allclose(current_params, new_params)
        assert current_params.requires_grad is True
    
    def test_get_circuit_info(self):
        """Test circuit information retrieval."""
        n_features, n_layers = 5, 3
        classifier = AngleEncodingClassifier(n_features, n_layers)
        
        info = classifier.get_circuit_info()
        
        expected_info = {
            "n_qubits": n_features,
            "n_layers": n_layers,
            "n_parameters": n_layers * n_features * 2,  # 3 * 5 * 2 = 30
            "encoding_type": "Angle Encoding (RY rotations)",
            "entanglement": "Linear CNOT chain",
            "variational_gates": "RY, RZ rotations"
        }
        
        assert info == expected_info
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the circuit."""
        classifier = AngleEncodingClassifier(n_features=3, n_layers=2, seed=42)
        
        features = torch.randn(3)
        
        # Enable gradient computation
        features.requires_grad_(True)
        
        prediction = classifier(features)
        prediction.backward()
        
        # Check that gradients exist
        assert features.grad is not None
        assert classifier.weights.grad is not None
    
    def test_different_architectures(self):
        """Test different circuit architectures."""
        test_cases = [
            (2, 1),   # Minimal case
            (8, 4),   # Larger case
            (16, 1),  # Wide, shallow
            (4, 8),   # Narrow, deep
        ]
        
        for n_features, n_layers in test_cases:
            classifier = AngleEncodingClassifier(n_features, n_layers)
            
            features = torch.randn(n_features)
            prediction = classifier(features)
            
            assert isinstance(prediction, torch.Tensor)
            assert prediction.shape == torch.Size([])
            assert -1 <= prediction <= 1
            
            # Check parameter count
            expected_params = n_layers * n_features * 2
            assert classifier.weights.numel() == expected_params


class TestIntegration:
    """Integration tests for the complete Angle Encoding system."""
    
    def test_training_compatibility(self):
        """Test compatibility with PyTorch training loops."""
        # Create model
        classifier = AngleEncodingClassifier(n_features=4, n_layers=2, seed=42)
        
        # Create dummy data
        X = torch.randn(10, 4)  # 10 samples, 4 features
        y = torch.randint(0, 2, (10,)).float() * 2 - 1  # Binary labels {-1, +1}
        
        # Define loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([classifier.weights], lr=0.1)
        
        # Training step
        initial_loss = float('inf')
        for epoch in range(3):
            optimizer.zero_grad()
            
            # Forward pass (batch processing)
            predictions = classifier(X)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        # Check that loss decreased (model is learning)
        final_loss = loss.item()
        assert final_loss < initial_loss, "Model should be learning"
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        n_features, n_layers = 4, 2
        features = torch.randn(n_features)
        
        # Create two identical models
        classifier1 = AngleEncodingClassifier(n_features, n_layers, seed=42)
        classifier2 = AngleEncodingClassifier(n_features, n_layers, seed=42)
        
        # Get predictions
        pred1 = classifier1(features)
        pred2 = classifier2(features)
        
        assert torch.allclose(pred1, pred2, atol=1e-6)
    
    def test_circuit_depth_scaling(self):
        """Test that deeper circuits don't break."""
        n_features = 4
        
        # Test various depths
        for n_layers in [1, 2, 5, 10]:
            classifier = AngleEncodingClassifier(n_features, n_layers)
            features = torch.randn(n_features)
            
            prediction = classifier(features)
            
            assert isinstance(prediction, torch.Tensor)
            assert -1 <= prediction <= 1
            
            # Deeper circuits should have more parameters
            expected_params = n_layers * n_features * 2
            assert classifier.weights.numel() == expected_params


# Utility test functions
def test_pennylane_available():
    """Test that PennyLane is properly installed."""
    import pennylane as qml
    
    # Test basic device creation
    dev = qml.device("default.qubit", wires=2)
    # Check device has expected properties (newer PennyLane API)
    assert hasattr(dev, 'wires') or hasattr(dev, 'num_wires')


def test_torch_integration():
    """Test PyTorch integration."""
    # Test that torch tensors work with PennyLane
    device = qml.device("default.qubit", wires=2)
    
    @qml.qnode(device, interface="torch")
    def test_circuit(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    x = torch.tensor(0.5, requires_grad=True)
    result = test_circuit(x)
    result.backward()
    
    assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])