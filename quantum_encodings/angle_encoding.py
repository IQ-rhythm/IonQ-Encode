"""
Angle Encoding (AE) implementation for quantum machine learning.

This module implements angle encoding circuits with RY rotations for data encoding
and variational layers for classification tasks. The encoding maps classical data
to quantum states through rotation angles.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional


def build_ae_classifier(n_qubits: int, n_layers: int) -> qml.QNode:
    """
    Build an Angle Encoding quantum classifier with variational layers.
    
    This function constructs a quantum neural network that:
    1. Encodes input features using RY rotations (angle encoding)
    2. Adds entangling CNOT chains for quantum correlations
    3. Applies variational layers with RY and RZ rotations
    4. Returns logit from the Z expectation value of the last qubit
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
            Should match the dimensionality of input features.
        n_layers (int): Number of variational layers to apply.
            Each layer contains RY, RZ rotations and CNOT entangling gates.
    
    Returns:
        qml.QNode: A quantum node function that can be called with inputs
            (features, weights) and returns a scalar prediction.
    
    Raises:
        ValueError: If n_qubits or n_layers are not positive integers.
    
    Example:
        >>> device = qml.device("default.qubit", wires=4)
        >>> classifier = build_ae_classifier(n_qubits=4, n_layers=2)
        >>> features = torch.randn(4)
        >>> weights = torch.randn(get_ae_weights_shape(4, 2))
        >>> prediction = classifier(features, weights)
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")
    if n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def ae_circuit(features: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Angle encoding quantum circuit with variational layers.
        
        Args:
            features (torch.Tensor): Input features of shape (n_qubits,).
                Each feature value becomes a rotation angle for RY gates.
            weights (torch.Tensor): Variational parameters of shape 
                (n_layers, n_qubits, 2). The last dimension contains
                [RY_angle, RZ_angle] for each qubit in each layer.
        
        Returns:
            float: Expectation value of Pauli-Z on the last qubit,
                used as the classification logit.
        """
        # Step 1: Encode input features with RY rotations (Angle Encoding)
        encode_features(features)
        
        # Step 2: Apply entangling CNOT chain
        apply_entangling_layer(n_qubits)
        
        # Step 3: Apply variational layers
        for layer in range(n_layers):
            apply_variational_layer(weights[layer], n_qubits)
            if layer < n_layers - 1:  # No entanglement after the last layer
                apply_entangling_layer(n_qubits)
        
        # Step 4: Return logit from Z expectation of last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return ae_circuit


def encode_features(features: torch.Tensor) -> None:
    """
    Encode classical features into quantum states using RY rotations.
    
    Each feature value is used as a rotation angle for an RY gate,
    mapping classical data to quantum amplitudes via:
    |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    
    Args:
        features (torch.Tensor): Input features of shape (n_features,).
            Each value represents a rotation angle in radians.
    """
    for i, feature in enumerate(features):
        qml.RY(feature, wires=i)


def apply_entangling_layer(n_qubits: int) -> None:
    """
    Apply a linear chain of CNOT gates for qubit entanglement.
    
    Creates quantum correlations between adjacent qubits using the pattern:
    CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)
    
    This linear topology is well-suited for near-term quantum devices
    with limited connectivity, especially ion trap architectures like IonQ.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
    """
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def apply_variational_layer(layer_weights: torch.Tensor, n_qubits: int) -> None:
    """
    Apply a single variational layer with RY and RZ rotations.
    
    Each variational layer applies parameterized rotations to all qubits:
    - RY rotation for amplitude manipulation
    - RZ rotation for phase manipulation
    
    This combination provides universal single-qubit gate coverage,
    enabling the circuit to approximate arbitrary unitary transformations
    when combined with entangling gates.
    
    Args:
        layer_weights (torch.Tensor): Parameters for this layer of shape
            (n_qubits, 2). The last dimension contains [RY_angle, RZ_angle]
            for each qubit.
        n_qubits (int): Number of qubits in the circuit.
    """
    for i in range(n_qubits):
        qml.RY(layer_weights[i, 0], wires=i)  # Amplitude rotation
        qml.RZ(layer_weights[i, 1], wires=i)  # Phase rotation


def get_ae_weights_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    """
    Get the required shape for weight parameters in the AE classifier.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of variational layers.
    
    Returns:
        Tuple[int, int, int]: Shape tuple (n_layers, n_qubits, 2).
            The dimensions represent [layer_index, qubit_index, rotation_type].
    
    Example:
        >>> shape = get_ae_weights_shape(n_qubits=4, n_layers=2)
        >>> weights = torch.randn(shape)
        >>> print(weights.shape)  # torch.Size([2, 4, 2])
    """
    return (n_layers, n_qubits, 2)


def initialize_ae_weights(n_qubits: int, n_layers: int, 
                         seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize weights for the Angle Encoding classifier.
    
    Weights are initialized using a normal distribution scaled by π/4
    to provide reasonable initial rotation angles for quantum gates.
    
    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of variational layers.
        seed (Optional[int]): Random seed for reproducible initialization.
    
    Returns:
        torch.Tensor: Initialized weights of shape (n_layers, n_qubits, 2).
    
    Example:
        >>> weights = initialize_ae_weights(n_qubits=4, n_layers=2, seed=42)
        >>> print(weights.shape)  # torch.Size([2, 4, 2])
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    shape = get_ae_weights_shape(n_qubits, n_layers)
    # Initialize with small random values scaled by π/4
    weights = torch.randn(shape) * (np.pi / 4)
    weights.requires_grad_(True)
    
    return weights


def create_ae_model(n_features: int, n_layers: int = 2) -> Tuple[qml.QNode, torch.Tensor]:
    """
    Create a complete Angle Encoding model with initialized weights.
    
    This is a convenience function that combines circuit creation and
    weight initialization for quick model setup.
    
    Args:
        n_features (int): Number of input features (determines n_qubits).
        n_layers (int, optional): Number of variational layers. Defaults to 2.
    
    Returns:
        Tuple[qml.QNode, torch.Tensor]: A tuple containing:
            - The quantum circuit function
            - Initialized weight parameters
    
    Example:
        >>> model, weights = create_ae_model(n_features=4, n_layers=3)
        >>> features = torch.randn(4)
        >>> prediction = model(features, weights)
    """
    circuit = build_ae_classifier(n_qubits=n_features, n_layers=n_layers)
    weights = initialize_ae_weights(n_qubits=n_features, n_layers=n_layers)
    
    return circuit, weights


class AngleEncodingClassifier:
    """
    A PyTorch-compatible wrapper for the Angle Encoding quantum classifier.
    
    This class provides a scikit-learn-like interface for the quantum circuit,
    making it easier to integrate with existing machine learning pipelines.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of variational layers.
        circuit (qml.QNode): The quantum circuit function.
        weights (torch.Tensor): Trainable parameters.
    """
    
    def __init__(self, n_features: int, n_layers: int = 2, seed: Optional[int] = None):
        """
        Initialize the Angle Encoding classifier.
        
        Args:
            n_features (int): Number of input features.
            n_layers (int, optional): Number of variational layers. Defaults to 2.
            seed (Optional[int]): Random seed for weight initialization.
        """
        self.n_qubits = n_features
        self.n_layers = n_layers
        
        self.circuit = build_ae_classifier(n_qubits=n_features, n_layers=n_layers)
        self.weights = initialize_ae_weights(n_qubits=n_features, 
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
            "encoding_type": "Angle Encoding (RY rotations)",
            "entanglement": "Linear CNOT chain",
            "variational_gates": "RY, RZ rotations"
        }
