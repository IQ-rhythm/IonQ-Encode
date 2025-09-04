"""
Amplitude Encoding (AE) implementation for quantum machine learning.

This module implements amplitude encoding circuits where classical data is
directly encoded into the amplitudes of quantum states using both exact
and approximate methods.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional, Union
import math


def normalize_vector(vector: Union[torch.Tensor, np.ndarray], 
                    method: str = "l2") -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize input vector for amplitude encoding.
    
    Args:
        vector: Input vector to normalize
        method: Normalization method ("l2" for L2 norm, "unit" for unit sum)
    
    Returns:
        Normalized vector with same type as input
    """
    if method == "l2":
        if isinstance(vector, torch.Tensor):
            norm = torch.norm(vector)
            return vector / norm if norm > 1e-10 else vector
        else:
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 1e-10 else vector
    elif method == "unit":
        if isinstance(vector, torch.Tensor):
            sum_abs = torch.sum(torch.abs(vector))
            return vector / sum_abs if sum_abs > 1e-10 else vector
        else:
            sum_abs = np.sum(np.abs(vector))
            return vector / sum_abs if sum_abs > 1e-10 else vector
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pad_or_truncate_vector(vector: Union[torch.Tensor, np.ndarray], 
                          target_size: int) -> Union[torch.Tensor, np.ndarray]:
    """
    Handle dimensional mismatch by padding with zeros or truncating.
    
    Args:
        vector: Input vector
        target_size: Target dimension (must be power of 2 for amplitude encoding)
    
    Returns:
        Vector with target_size dimension
    """
    current_size = len(vector)
    
    if current_size == target_size:
        return vector
    elif current_size < target_size:
        # Pad with zeros
        if isinstance(vector, torch.Tensor):
            padding = torch.zeros(target_size - current_size, dtype=vector.dtype)
            return torch.cat([vector, padding])
        else:
            return np.concatenate([vector, np.zeros(target_size - current_size)])
    else:
        # Truncate
        return vector[:target_size]


def validate_amplitude_encoding_size(n_features: int) -> int:
    """
    Validate and determine the number of qubits needed for amplitude encoding.
    
    Args:
        n_features: Number of input features
    
    Returns:
        Number of qubits (log2 of next power of 2)
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    
    # Find next power of 2
    n_qubits = math.ceil(math.log2(n_features))
    return n_qubits


def build_exact_amplitude_classifier(n_qubits: int, n_layers: int = 1) -> qml.QNode:
    """
    Build an exact amplitude encoding classifier using MottonenStatePreparation.
    
    Args:
        n_qubits: Number of qubits (determines 2^n_qubits amplitudes)
        n_layers: Number of variational layers (default 1)
    
    Returns:
        Quantum circuit function for exact amplitude encoding
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def amplitude_circuit(features: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Exact amplitude encoding circuit.
        
        Args:
            features: Input features (will be normalized and padded/truncated)
            weights: Variational parameters for post-encoding layers
        
        Returns:
            Binary classification logit from PauliZ measurement
        """
        # Prepare features for amplitude encoding
        target_size = 2 ** n_qubits
        processed_features = pad_or_truncate_vector(features, target_size)
        normalized_features = normalize_vector(processed_features, method="l2")
        
        # Exact amplitude encoding using MottonenStatePreparation
        qml.MottonenStatePreparation(normalized_features, wires=range(n_qubits))
        
        # Apply variational layers
        for layer in range(n_layers):
            apply_variational_layer(weights[layer], n_qubits)
            if layer < n_layers - 1:
                apply_entangling_layer(n_qubits)
        
        # Binary classifier readout using PauliZ on last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return amplitude_circuit


def build_approximate_amplitude_classifier(n_qubits: int, n_layers: int = 1) -> qml.QNode:
    """
    Build an approximate amplitude encoding classifier (EnQode-style stub).
    
    This is a placeholder for more advanced compression techniques.
    Currently implements a simplified version using angle encoding as approximation.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
    
    Returns:
        Quantum circuit function for approximate amplitude encoding
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def approx_amplitude_circuit(features: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Approximate amplitude encoding circuit (stub implementation).
        
        Args:
            features: Input features
            weights: Variational parameters
        
        Returns:
            Binary classification logit from PauliZ measurement
        """
        # Approximate encoding: use first n_qubits features for angle encoding
        processed_features = pad_or_truncate_vector(features, n_qubits)
        normalized_features = normalize_vector(processed_features, method="l2")
        
        # Approximate amplitude encoding using RY rotations (placeholder)
        for i in range(n_qubits):
            qml.RY(normalized_features[i] * np.pi, wires=i)
        
        # Apply variational layers
        for layer in range(n_layers):
            apply_variational_layer(weights[layer], n_qubits)
            if layer < n_layers - 1:
                apply_entangling_layer(n_qubits)
        
        # Binary classifier readout using PauliZ on last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return approx_amplitude_circuit


def apply_entangling_layer(n_qubits: int) -> None:
    """
    Apply a linear chain of CNOT gates for qubit entanglement.
    
    Args:
        n_qubits: Number of qubits in the circuit
    """
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def apply_variational_layer(layer_weights: torch.Tensor, n_qubits: int) -> None:
    """
    Apply a single variational layer with RY and RZ rotations.
    
    Args:
        layer_weights: Parameters for this layer of shape (n_qubits, 2)
        n_qubits: Number of qubits in the circuit
    """
    for i in range(n_qubits):
        qml.RY(layer_weights[i, 0], wires=i)
        qml.RZ(layer_weights[i, 1], wires=i)


def get_amplitude_weights_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    """
    Get the required shape for weight parameters in amplitude encoding classifiers.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
    
    Returns:
        Shape tuple (n_layers, n_qubits, 2)
    """
    return (n_layers, n_qubits, 2)


def initialize_amplitude_weights(n_qubits: int, n_layers: int, 
                                seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize weights for amplitude encoding classifiers.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
        seed: Random seed for reproducible initialization
    
    Returns:
        Initialized weights of shape (n_layers, n_qubits, 2)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    shape = get_amplitude_weights_shape(n_qubits, n_layers)
    weights = torch.randn(shape) * (np.pi / 4)
    weights.requires_grad_(True)
    
    return weights


def create_amplitude_model(n_features: int, n_layers: int = 1, 
                          method: str = "exact") -> Tuple[qml.QNode, torch.Tensor]:
    """
    Create a complete amplitude encoding model with initialized weights.
    
    Args:
        n_features: Number of input features
        n_layers: Number of variational layers
        method: Encoding method ("exact" or "approximate")
    
    Returns:
        Tuple of (quantum circuit function, initialized weights)
    """
    n_qubits = validate_amplitude_encoding_size(n_features)
    
    if method == "exact":
        circuit = build_exact_amplitude_classifier(n_qubits, n_layers)
    elif method == "approximate":
        circuit = build_approximate_amplitude_classifier(n_qubits, n_layers)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'approximate'")
    
    weights = initialize_amplitude_weights(n_qubits, n_layers)
    
    return circuit, weights


class AmplitudeEncodingClassifier:
    """
    A PyTorch-compatible wrapper for amplitude encoding quantum classifiers.
    
    This class provides a scikit-learn-like interface for quantum amplitude
    encoding circuits with both exact and approximate methods.
    """
    
    def __init__(self, n_features: int, n_layers: int = 1, 
                 method: str = "exact", seed: Optional[int] = None):
        """
        Initialize the amplitude encoding classifier.
        
        Args:
            n_features: Number of input features
            n_layers: Number of variational layers
            method: Encoding method ("exact" or "approximate")
            seed: Random seed for weight initialization
        """
        self.n_features = n_features
        self.n_qubits = validate_amplitude_encoding_size(n_features)
        self.n_layers = n_layers
        self.method = method
        
        if method == "exact":
            self.circuit = build_exact_amplitude_classifier(self.n_qubits, n_layers)
        elif method == "approximate":
            self.circuit = build_approximate_amplitude_classifier(self.n_qubits, n_layers)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.weights = initialize_amplitude_weights(self.n_qubits, n_layers, seed)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        
        Args:
            features: Input features of shape (batch_size, n_features) or (n_features,)
        
        Returns:
            Predictions of shape (batch_size,) or scalar for single samples
        """
        if len(features.shape) == 1:
            return self.circuit(features, self.weights)
        else:
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
            Dictionary containing circuit specifications
        """
        return {
            "n_features": self.n_features,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.weights.numel(),
            "encoding_type": f"Amplitude Encoding ({self.method})",
            "entanglement": "Linear CNOT chain",
            "variational_gates": "RY, RZ rotations",
            "readout": "PauliZ on last qubit"
        }