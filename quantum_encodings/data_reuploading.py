"""
Data Re-Uploading (DRU) implementation for quantum machine learning.

This module implements data re-uploading circuits where classical data is
repeatedly injected into the quantum circuit through multiple layers,
combined with variational parameters and entanglement operations.
"""

import pennylane as qml
import numpy as np
from typing import Optional


def build_dru_classifier(n_qubits: int, n_layers: int) -> qml.QNode:
    """
    Build a Data Re-Uploading quantum classifier.
    
    This function constructs a quantum neural network that:
    1. Repeatedly injects input data through RY rotations in each layer
    2. Applies variational parameters as RY rotations
    3. Adds entangling CNOT gates between layers
    4. Returns logit from the Z expectation value of the last qubit
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of re-uploading layers to apply.
    
    Returns:
        qml.QNode: A quantum node function that can be called with inputs
            (features, weights) and returns a scalar prediction.
    
    Example:
        >>> circuit = build_dru_classifier(n_qubits=3, n_layers=2)
        >>> inputs = np.array([0.5, 1.0, 1.5])
        >>> weights = initialize_dru_weights(3, 2)
        >>> prediction = circuit(inputs, weights)
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")
    if n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="autograd")
    def dru_circuit(inputs, weights):
        """
        Data Re-Uploading quantum circuit.
        
        Args:
            inputs: Input features to be re-uploaded in each layer
            weights: Variational parameters of shape (n_layers, n_qubits)
        
        Returns:
            float: Expectation value of Pauli-Z on the last qubit
        """
        # Apply DRU layers
        for layer in range(n_layers):
            _dru_layer(inputs, weights[layer], n_qubits, layer)
        
        # Return Z expectation of last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return dru_circuit


def _dru_layer(inputs, layer_weights, n_qubits: int, layer_idx: int) -> None:
    """
    Apply a single Data Re-Uploading layer.
    
    Each DRU layer consists of:
    1. Data injection through RY rotations
    2. Variational parameters as RY rotations  
    3. Entangling CNOT gates (except for the last layer)
    
    Args:
        inputs: Input data features
        layer_weights: Variational parameters for this layer
        n_qubits (int): Number of qubits
        layer_idx (int): Index of current layer
    """
    # Data injection - re-upload input features
    for i in range(min(len(inputs), n_qubits)):
        qml.RY(inputs[i], wires=i)
    
    # Variational parameters
    for i in range(n_qubits):
        qml.RY(layer_weights[i], wires=i)
    
    # Entangling layer - linear chain of CNOTs
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def initialize_dru_weights(n_qubits: int, n_layers: int, 
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize weights for the DRU classifier.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of DRU layers.
        seed (Optional[int]): Random seed for reproducible initialization.
    
    Returns:
        np.ndarray: Initialized weights of shape (n_layers, n_qubits).
    
    Example:
        >>> weights = initialize_dru_weights(n_qubits=4, n_layers=2, seed=42)
        >>> print(weights.shape)  # (2, 4)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize weights in [0, 2π] range
    weights = np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits))
    
    return weights


def dru_feature_dimension(n_qubits: int) -> int:
    """
    Get the feature dimension for DRU encoding.
    
    In DRU, the feature dimension matches the number of qubits
    since data is re-uploaded to each qubit in each layer.
    
    Args:
        n_qubits (int): Number of qubits
    
    Returns:
        int: Required feature dimension
    """
    return n_qubits