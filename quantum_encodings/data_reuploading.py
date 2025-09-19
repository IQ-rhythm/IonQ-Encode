"""
Data Re-Uploading (DRU) implementation for quantum machine learning.

This module implements data re-uploading circuits where classical data is
repeatedly injected into the quantum circuit through multiple layers,
combined with variational parameters and entanglement operations.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Optional, Tuple


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
    
    @qml.qnode(device, interface="torch")
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
    # Validate weight dimensions
    if len(layer_weights) != n_qubits:
        raise IndexError(f"Expected {n_qubits} weights, got {len(layer_weights)}")
    
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


def get_dru_weights_shape(n_qubits: int, n_layers: int) -> Tuple[int, int]:
    """
    Get the required shape for weight parameters in the DRU classifier.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of DRU layers.
    
    Returns:
        Tuple[int, int]: Shape tuple (n_layers, n_qubits).
    
    Example:
        >>> shape = get_dru_weights_shape(n_qubits=4, n_layers=2)
        >>> weights = torch.randn(shape)
        >>> print(weights.shape)  # torch.Size([2, 4])
    """
    return (n_layers, n_qubits)


def initialize_dru_weights_torch(n_qubits: int, n_layers: int, 
                                seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize PyTorch weights for the DRU classifier.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of DRU layers.
        seed (Optional[int]): Random seed for reproducible initialization.
    
    Returns:
        torch.Tensor: Initialized weights of shape (n_layers, n_qubits).
    
    Example:
        >>> weights = initialize_dru_weights_torch(n_qubits=4, n_layers=2, seed=42)
        >>> print(weights.shape)  # torch.Size([2, 4])
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    shape = get_dru_weights_shape(n_qubits, n_layers)
    # Initialize weights in [0, 2π] range
    weights = torch.rand(shape) * (2 * np.pi)
    weights.requires_grad_(True)
    
    return weights


def create_dru_model(n_features: int, n_layers: int = 2) -> Tuple[qml.QNode, torch.Tensor]:
    """
    Create a complete Data Re-Uploading model with initialized weights.
    
    This is a convenience function that combines circuit creation and
    weight initialization for quick model setup.
    
    Args:
        n_features (int): Number of input features (determines n_qubits).
        n_layers (int, optional): Number of DRU layers. Defaults to 2.
    
    Returns:
        Tuple[qml.QNode, torch.Tensor]: A tuple containing:
            - The quantum circuit function
            - Initialized weight parameters
    
    Example:
        >>> model, weights = create_dru_model(n_features=4, n_layers=3)
        >>> features = torch.randn(4)
        >>> prediction = model(features, weights)
    """
    circuit = build_dru_classifier(n_qubits=n_features, n_layers=n_layers)
    weights = initialize_dru_weights_torch(n_qubits=n_features, n_layers=n_layers)
    
    return circuit, weights


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


class DRUClassifier:
    """
    A PyTorch-compatible wrapper for the Data Re-Uploading quantum classifier.
    
    This class provides a scikit-learn-like interface for the quantum circuit,
    making it easier to integrate with existing machine learning pipelines.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of DRU layers.
        circuit (qml.QNode): The quantum circuit function.
        weights (torch.Tensor): Trainable parameters.
    """
    
    def __init__(self, n_features: int, n_layers: int = 2, seed: Optional[int] = None):
        """
        Initialize the Data Re-Uploading classifier.
        
        Args:
            n_features (int): Number of input features.
            n_layers (int, optional): Number of DRU layers. Defaults to 2.
            seed (Optional[int]): Random seed for weight initialization.
        """
        self.n_qubits = n_features
        self.n_layers = n_layers
        
        self.circuit = build_dru_classifier(n_qubits=n_features, n_layers=n_layers)
        self.weights = initialize_dru_weights_torch(n_qubits=n_features, 
                                                  n_layers=n_layers, seed=seed)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, n_features)
                or (n_features,) for single samples.
        
        Returns:
            torch.Tensor: Predictions of shape (batch_size,) or scalar for single samples.
        """
        if len(features.shape) == 1:
            # Single sample
            return self.circuit(features, self.weights)
        else:
            # Batch processing
            batch_size = features.shape[0]
            predictions = torch.zeros(batch_size)
            
            for i in range(batch_size):
                predictions[i] = self.circuit(features[i], self.weights)
            
            return predictions
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.forward(features)
    
    def get_params(self) -> torch.Tensor:
        """Get the current weight parameters."""
        return self.weights.detach().clone()
    
    def set_params(self, weights: torch.Tensor) -> None:
        """Set the weight parameters."""
        self.weights = weights.requires_grad_(True)
    
    def get_circuit_info(self) -> dict:
        """
        Get information about the quantum circuit.
        
        Returns:
            dict: Dictionary containing circuit specifications.
        """
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.weights.numel(),
            "encoding_type": "Data Re-Uploading (repeated injection)",
            "entanglement": "Linear CNOT chain",
            "variational_gates": "RY rotations"
        }