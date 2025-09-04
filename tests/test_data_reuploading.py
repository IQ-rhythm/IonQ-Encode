"""
Unit tests for Data Re-Uploading (DRU) quantum encoding implementation.
"""

import pytest
import numpy as np
import pennylane as qml
from quantum_encodings.data_reuploading import (
    build_dru_classifier,
    initialize_dru_weights,
    dru_feature_dimension,
    _dru_layer
)


class TestDRUClassifier:
    """Test suite for DRU classifier functionality."""
    
    def test_build_dru_classifier_basic(self):
        """Test basic DRU classifier construction."""
        n_qubits, n_layers = 4, 2
        circuit = build_dru_classifier(n_qubits, n_layers)
        assert callable(circuit)
    
    def test_dru_circuit_execution(self):
        """Test DRU circuit executes and returns valid output."""
        n_qubits, n_layers = 3, 2
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        inputs = np.array([0.5, 1.0, 1.5])
        weights = initialize_dru_weights(n_qubits, n_layers)
        
        result = circuit(inputs, weights)
        assert isinstance(result, (float, np.ndarray))
        assert -1 <= result <= 1  # Z expectation bounds
    
    def test_dru_output_deterministic(self):
        """Test DRU circuit produces consistent results."""
        n_qubits, n_layers = 2, 1
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        inputs = np.array([0.3, 0.7])
        weights = np.array([[0.1, 0.2]])
        
        result1 = circuit(inputs, weights)
        result2 = circuit(inputs, weights)
        np.testing.assert_almost_equal(result1, result2, decimal=10)
    
    def test_dru_different_inputs_different_outputs(self):
        """Test DRU produces different outputs for different inputs."""
        n_qubits, n_layers = 3, 1
        circuit = build_dru_classifier(n_qubits, n_layers)
        weights = np.array([[0.5, 0.5, 0.5]])
        
        inputs1 = np.array([0.0, 0.0, 0.0])
        inputs2 = np.array([1.0, 1.0, 1.0])
        
        result1 = circuit(inputs1, weights)
        result2 = circuit(inputs2, weights)
        assert abs(result1 - result2) > 1e-6
    
    def test_dru_weight_sensitivity(self):
        """Test DRU is sensitive to weight changes."""
        n_qubits, n_layers = 2, 1
        circuit = build_dru_classifier(n_qubits, n_layers)
        inputs = np.array([0.5, 0.8])
        
        weights1 = np.array([[0.1, 0.2]])
        weights2 = np.array([[0.9, 0.8]])
        
        result1 = circuit(inputs, weights1)
        result2 = circuit(inputs, weights2)
        assert abs(result1 - result2) > 1e-6


class TestDRULayers:
    """Test suite for DRU layer scaling."""
    
    def test_single_layer_execution(self):
        """Test single layer DRU execution."""
        n_qubits, n_layers = 2, 1
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        inputs = np.array([0.2, 0.4])
        weights = np.array([[0.1, 0.3]])
        
        result = circuit(inputs, weights)
        assert isinstance(result, (float, np.ndarray))
    
    def test_multiple_layers_execution(self):
        """Test multi-layer DRU execution."""
        n_qubits, n_layers = 3, 4
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        inputs = np.array([0.1, 0.5, 0.9])
        weights = initialize_dru_weights(n_qubits, n_layers)
        
        result = circuit(inputs, weights)
        assert isinstance(result, (float, np.ndarray))
    
    def test_layer_depth_effect(self):
        """Test that different layer depths produce different results."""
        n_qubits = 2
        inputs = np.array([0.3, 0.7])
        
        # Single layer
        circuit1 = build_dru_classifier(n_qubits, 1)
        weights1 = np.array([[0.2, 0.4]])
        result1 = circuit1(inputs, weights1)
        
        # Two layers with same per-layer weights
        circuit2 = build_dru_classifier(n_qubits, 2)
        weights2 = np.array([[0.2, 0.4], [0.2, 0.4]])
        result2 = circuit2(inputs, weights2)
        
        assert abs(result1 - result2) > 1e-6


class TestDRUUtilities:
    """Test suite for DRU utility functions."""
    
    def test_initialize_dru_weights_shape(self):
        """Test weight initialization produces correct shape."""
        n_qubits, n_layers = 5, 3
        weights = initialize_dru_weights(n_qubits, n_layers)
        assert weights.shape == (n_layers, n_qubits)
    
    def test_initialize_dru_weights_range(self):
        """Test weights are in valid range [0, 2π]."""
        weights = initialize_dru_weights(4, 2)
        assert np.all(weights >= 0)
        assert np.all(weights <= 2 * np.pi)
    
    def test_initialize_dru_weights_reproducible(self):
        """Test weight initialization is reproducible with same seed."""
        weights1 = initialize_dru_weights(3, 2, seed=123)
        weights2 = initialize_dru_weights(3, 2, seed=123)
        np.testing.assert_array_equal(weights1, weights2)
    
    def test_initialize_dru_weights_different_seeds(self):
        """Test different seeds produce different weights."""
        weights1 = initialize_dru_weights(3, 2, seed=1)
        weights2 = initialize_dru_weights(3, 2, seed=2)
        assert not np.array_equal(weights1, weights2)
    
    def test_dru_feature_dimension(self):
        """Test feature dimension calculation."""
        assert dru_feature_dimension(4) == 4
        assert dru_feature_dimension(8) == 8
        assert dru_feature_dimension(1) == 1


class TestDRUCircuitStructure:
    """Test suite for DRU circuit structure validation."""
    
    def test_circuit_qubit_count(self):
        """Test circuit uses correct number of qubits."""
        n_qubits = 3
        circuit = build_dru_classifier(n_qubits, 1)
        
        # Extract device from QNode
        device = circuit.device
        assert len(device.wires) == n_qubits
    
    def test_input_dimension_matching(self):
        """Test circuit handles input dimensions correctly."""
        n_qubits = 4
        circuit = build_dru_classifier(n_qubits, 1)
        
        # Test with matching input dimension
        inputs = np.random.rand(n_qubits)
        weights = initialize_dru_weights(n_qubits, 1)
        result = circuit(inputs, weights)
        assert isinstance(result, (float, np.ndarray))
        
        # Test with shorter input (should still work)
        inputs_short = np.random.rand(n_qubits - 1)
        result_short = circuit(inputs_short, weights)
        assert isinstance(result_short, (float, np.ndarray))
    
    def test_weight_dimension_validation(self):
        """Test circuit requires correct weight dimensions."""
        n_qubits, n_layers = 3, 2
        circuit = build_dru_classifier(n_qubits, n_layers)
        inputs = np.array([0.1, 0.2, 0.3])
        
        # Correct weights
        correct_weights = initialize_dru_weights(n_qubits, n_layers)
        result = circuit(inputs, correct_weights)
        assert isinstance(result, (float, np.ndarray))
        
        # Wrong weight shape should raise error
        wrong_weights = np.random.rand(n_layers, n_qubits + 1)
        with pytest.raises((IndexError, ValueError)):
            circuit(inputs, wrong_weights)


class TestDRUMathematicalProperties:
    """Test suite for mathematical properties of DRU."""
    
    def test_expectation_value_bounds(self):
        """Test Z expectation values are properly bounded."""
        n_qubits, n_layers = 4, 3
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        for _ in range(10):  # Test multiple random cases
            inputs = np.random.uniform(-2, 2, n_qubits)
            weights = initialize_dru_weights(n_qubits, n_layers)
            result = circuit(inputs, weights)
            
            assert -1 <= result <= 1
    
    def test_zero_input_behavior(self):
        """Test circuit behavior with zero inputs."""
        n_qubits, n_layers = 3, 2
        circuit = build_dru_classifier(n_qubits, n_layers)
        
        zero_inputs = np.zeros(n_qubits)
        weights = initialize_dru_weights(n_qubits, n_layers)
        result = circuit(zero_inputs, weights)
        
        assert isinstance(result, (float, np.ndarray))
        assert -1 <= result <= 1
    
    def test_scaling_invariance_properties(self):
        """Test how DRU responds to input scaling."""
        n_qubits, n_layers = 2, 1
        circuit = build_dru_classifier(n_qubits, n_layers)
        weights = np.array([[0.5, 0.5]])
        
        base_inputs = np.array([0.1, 0.2])
        scaled_inputs = base_inputs * 2
        
        result_base = circuit(base_inputs, weights)
        result_scaled = circuit(scaled_inputs, weights)
        
        # Results should be different (DRU is not scale-invariant)
        assert abs(result_base - result_scaled) > 1e-10


if __name__ == "__main__":
    pytest.main([__file__])