"""
Hybrid Encoding implementation for quantum machine learning.

This module implements hybrid encoding circuits that combine both angle encoding (AE)
and amplitude encoding (AMP) methods for richer data representation and improved
expressivity in quantum machine learning models.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional, Union, Dict
import math

from .amplitude_encoding import normalize_vector, pad_or_truncate_vector
from .angle_encoding import apply_entangling_layer as ae_entangling_layer


def split_features_for_hybrid(features: Union[torch.Tensor, np.ndarray], 
                             n_angle_features: int, 
                             n_amplitude_features: int) -> Tuple[Union[torch.Tensor, np.ndarray], 
                                                               Union[torch.Tensor, np.ndarray]]:
    """
    Split input features for hybrid encoding into angle and amplitude components.
    
    Args:
        features: Input feature vector
        n_angle_features: Number of features to use for angle encoding
        n_amplitude_features: Number of features to use for amplitude encoding
    
    Returns:
        Tuple of (angle_features, amplitude_features)
    """
    total_needed = n_angle_features + n_amplitude_features
    
    if len(features) < total_needed:
        # Pad with zeros if needed
        if isinstance(features, torch.Tensor):
            features = torch.cat([features, torch.zeros(total_needed - len(features))])
        else:
            features = np.concatenate([features, np.zeros(total_needed - len(features))])
    elif len(features) > total_needed:
        # Truncate if needed
        features = features[:total_needed]
    
    angle_features = features[:n_angle_features]
    amplitude_features = features[n_angle_features:n_angle_features + n_amplitude_features]
    
    return angle_features, amplitude_features


def validate_hybrid_encoding_params(n_angle_qubits: int, n_amplitude_qubits: int) -> None:
    """
    Validate hybrid encoding parameters.
    
    Args:
        n_angle_qubits: Number of qubits for angle encoding
        n_amplitude_qubits: Number of qubits for amplitude encoding
    
    Raises:
        ValueError: If parameters are invalid
    """
    if n_angle_qubits <= 0:
        raise ValueError("n_angle_qubits must be positive")
    if n_amplitude_qubits <= 0:
        raise ValueError("n_amplitude_qubits must be positive")
    if n_angle_qubits + n_amplitude_qubits > 20:  # Practical limit
        raise ValueError("Total qubits should not exceed 20 for practical simulation")


def build_hybrid_classifier(n_angle_qubits: int, n_amplitude_qubits: int, 
                           n_layers: int = 2, entanglement_strategy: str = "linear") -> qml.QNode:
    """
    Build a hybrid encoding quantum classifier combining angle and amplitude encoding.
    
    The hybrid approach:
    1. Uses first n_angle_qubits for angle encoding (RY rotations)
    2. Uses next n_amplitude_qubits for amplitude encoding (MottonenStatePreparation)
    3. Applies cross-entanglement between the two encoding regions
    4. Adds variational layers for classification
    
    Args:
        n_angle_qubits: Number of qubits for angle encoding section
        n_amplitude_qubits: Number of qubits for amplitude encoding section  
        n_layers: Number of variational layers
        entanglement_strategy: Strategy for entanglement ("linear", "circular", "full")
    
    Returns:
        Quantum circuit function for hybrid encoding
    """
    validate_hybrid_encoding_params(n_angle_qubits, n_amplitude_qubits)
    
    total_qubits = n_angle_qubits + n_amplitude_qubits
    device = qml.device("default.qubit", wires=total_qubits)
    
    @qml.qnode(device, interface="torch")
    def hybrid_circuit(features: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Hybrid encoding quantum circuit.
        
        Args:
            features: Input features (will be split for angle and amplitude encoding)
            weights: Variational parameters
        
        Returns:
            Binary classification logit from PauliZ measurement
        """
        n_amplitude_features = 2 ** n_amplitude_qubits
        
        # Split features for hybrid encoding
        angle_features, amplitude_features = split_features_for_hybrid(
            features, n_angle_qubits, n_amplitude_features
        )
        
        # Step 1: Angle encoding on first n_angle_qubits
        for i in range(n_angle_qubits):
            qml.RY(angle_features[i], wires=i)
        
        # Step 2: Amplitude encoding on next n_amplitude_qubits
        normalized_amp_features = normalize_vector(amplitude_features, method="l2")
        amplitude_wires = list(range(n_angle_qubits, total_qubits))
        qml.MottonenStatePreparation(normalized_amp_features, wires=amplitude_wires)
        
        # Step 3: Cross-entanglement between encoding regions
        apply_cross_entanglement(n_angle_qubits, n_amplitude_qubits, entanglement_strategy)
        
        # Step 4: Variational layers
        for layer in range(n_layers):
            apply_hybrid_variational_layer(weights[layer], total_qubits)
            if layer < n_layers - 1:
                apply_hybrid_entanglement(total_qubits, entanglement_strategy)
        
        # Step 5: Readout from last qubit
        return qml.expval(qml.PauliZ(total_qubits - 1))
    
    return hybrid_circuit


def apply_cross_entanglement(n_angle_qubits: int, n_amplitude_qubits: int, 
                           strategy: str = "linear") -> None:
    """
    Apply cross-entanglement between angle and amplitude encoding regions.
    
    Args:
        n_angle_qubits: Number of qubits in angle encoding region
        n_amplitude_qubits: Number of qubits in amplitude encoding region
        strategy: Entanglement strategy
    """
    if strategy == "linear":
        # Connect the boundary between regions
        qml.CNOT(wires=[n_angle_qubits - 1, n_angle_qubits])
        
        # Linear chain within each region
        for i in range(n_angle_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        for i in range(n_angle_qubits + 1, n_angle_qubits + n_amplitude_qubits):
            qml.CNOT(wires=[i - 1, i])
            
    elif strategy == "circular":
        # Circular entanglement within each region plus cross-connection
        for i in range(n_angle_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_angle_qubits])
        
        for i in range(n_angle_qubits, n_angle_qubits + n_amplitude_qubits):
            next_wire = n_angle_qubits + ((i - n_angle_qubits + 1) % n_amplitude_qubits)
            qml.CNOT(wires=[i, next_wire])
        
        # Cross-connection
        qml.CNOT(wires=[n_angle_qubits - 1, n_angle_qubits])
        
    elif strategy == "full":
        # Full entanglement between regions (expensive but expressive)
        total_qubits = n_angle_qubits + n_amplitude_qubits
        for i in range(total_qubits):
            for j in range(i + 1, min(i + 3, total_qubits)):  # Limit connectivity
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"Unknown entanglement strategy: {strategy}")


def apply_hybrid_variational_layer(layer_weights: torch.Tensor, n_qubits: int) -> None:
    """
    Apply a variational layer optimized for hybrid encoding.
    
    Uses RY and RZ rotations on all qubits with additional RX rotations
    for increased expressivity in the hybrid setting.
    
    Args:
        layer_weights: Parameters for this layer of shape (n_qubits, 3)
        n_qubits: Total number of qubits
    """
    for i in range(n_qubits):
        qml.RX(layer_weights[i, 0], wires=i)  # X rotation for additional expressivity
        qml.RY(layer_weights[i, 1], wires=i)  # Y rotation (amplitude)
        qml.RZ(layer_weights[i, 2], wires=i)  # Z rotation (phase)


def apply_hybrid_entanglement(n_qubits: int, strategy: str = "linear") -> None:
    """
    Apply entanglement layer for hybrid encoding.
    
    Args:
        n_qubits: Total number of qubits
        strategy: Entanglement strategy
    """
    if strategy == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif strategy == "circular":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif strategy == "full":
        for i in range(n_qubits):
            for j in range(i + 1, min(i + 3, n_qubits)):
                qml.CNOT(wires=[i, j])


def get_hybrid_weights_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    """
    Get the required shape for weight parameters in hybrid encoding.
    
    Args:
        n_qubits: Total number of qubits
        n_layers: Number of variational layers
    
    Returns:
        Shape tuple (n_layers, n_qubits, 3) - 3 rotations per qubit
    """
    return (n_layers, n_qubits, 3)


def initialize_hybrid_weights(n_qubits: int, n_layers: int, 
                             seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize weights for hybrid encoding classifier.
    
    Args:
        n_qubits: Total number of qubits
        n_layers: Number of variational layers
        seed: Random seed for reproducible initialization
    
    Returns:
        Initialized weights of shape (n_layers, n_qubits, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    shape = get_hybrid_weights_shape(n_qubits, n_layers)
    weights = torch.randn(shape) * (np.pi / 6)  # Smaller initial values for 3 rotations
    weights.requires_grad_(True)
    
    return weights


def benchmark_hybrid_circuit(n_angle_qubits: int, n_amplitude_qubits: int, 
                            n_layers: int = 2, entanglement_strategy: str = "linear") -> Dict[str, int]:
    """
    Benchmark the complexity of a hybrid encoding circuit.
    
    Args:
        n_angle_qubits: Number of qubits for angle encoding
        n_amplitude_qubits: Number of qubits for amplitude encoding
        n_layers: Number of variational layers
        entanglement_strategy: Entanglement strategy
    
    Returns:
        Dictionary with complexity metrics
    """
    total_qubits = n_angle_qubits + n_amplitude_qubits
    
    # Count gates for different components
    gates = {
        "angle_encoding_gates": n_angle_qubits,  # RY gates
        "amplitude_encoding_gates": 0,  # MottonenStatePreparation complexity
        "cross_entanglement_gates": 0,
        "variational_gates": n_layers * total_qubits * 3,  # RX, RY, RZ per qubit per layer
        "entanglement_gates": 0
    }
    
    # Approximate MottonenStatePreparation gate count
    n_amplitude_features = 2 ** n_amplitude_qubits
    if n_amplitude_features > 1:
        # Rough estimate: O(2^n) for n-qubit state preparation
        gates["amplitude_encoding_gates"] = 2 * (2 ** n_amplitude_qubits - 1)
    
    # Cross-entanglement gates
    if entanglement_strategy == "linear":
        gates["cross_entanglement_gates"] = total_qubits  # One CNOT per adjacent pair + boundary
    elif entanglement_strategy == "circular":
        gates["cross_entanglement_gates"] = total_qubits + 1  # Circular + cross-connection
    elif entanglement_strategy == "full":
        gates["cross_entanglement_gates"] = min(total_qubits * 3, total_qubits * (total_qubits - 1) // 2)
    
    # Entanglement gates in variational layers
    entanglement_per_layer = 0
    if entanglement_strategy == "linear":
        entanglement_per_layer = total_qubits - 1
    elif entanglement_strategy == "circular":
        entanglement_per_layer = total_qubits
    elif entanglement_strategy == "full":
        entanglement_per_layer = min(total_qubits * 2, total_qubits * (total_qubits - 1) // 2)
    
    gates["entanglement_gates"] = (n_layers - 1) * entanglement_per_layer if n_layers > 0 else 0
    
    # Calculate total metrics
    total_gates = sum(gates.values())
    
    # Circuit depth estimation (rough approximation)
    encoding_depth = max(1, math.ceil(math.log2(n_amplitude_features))) + 1  # Amplitude + angle
    cross_entanglement_depth = 1
    variational_depth = n_layers * 4  # 3 rotations + entanglement per layer
    
    circuit_depth = encoding_depth + cross_entanglement_depth + variational_depth
    
    return {
        "total_qubits": total_qubits,
        "total_gates": total_gates,
        "circuit_depth": circuit_depth,
        "angle_encoding_gates": gates["angle_encoding_gates"],
        "amplitude_encoding_gates": gates["amplitude_encoding_gates"],
        "cross_entanglement_gates": gates["cross_entanglement_gates"],
        "variational_gates": gates["variational_gates"],
        "entanglement_gates": gates["entanglement_gates"]
    }


def create_hybrid_model(n_angle_features: int, n_amplitude_features_log: int, 
                       n_layers: int = 2, entanglement_strategy: str = "linear",
                       seed: Optional[int] = None) -> Tuple[qml.QNode, torch.Tensor, Dict[str, int]]:
    """
    Create a complete hybrid encoding model with initialized weights and benchmarking.
    
    Args:
        n_angle_features: Number of features for angle encoding (= n_angle_qubits)
        n_amplitude_features_log: Log2 of amplitude features (n_amplitude_qubits)
        n_layers: Number of variational layers
        entanglement_strategy: Entanglement strategy
        seed: Random seed for weight initialization
    
    Returns:
        Tuple of (circuit, weights, complexity_metrics)
    """
    n_angle_qubits = n_angle_features
    n_amplitude_qubits = n_amplitude_features_log
    total_qubits = n_angle_qubits + n_amplitude_qubits
    
    circuit = build_hybrid_classifier(n_angle_qubits, n_amplitude_qubits, 
                                     n_layers, entanglement_strategy)
    weights = initialize_hybrid_weights(total_qubits, n_layers, seed)
    complexity = benchmark_hybrid_circuit(n_angle_qubits, n_amplitude_qubits, 
                                        n_layers, entanglement_strategy)
    
    return circuit, weights, complexity


class HybridEncodingClassifier:
    """
    A PyTorch-compatible wrapper for hybrid encoding quantum classifiers.
    
    This class combines angle encoding and amplitude encoding for richer
    data representation and improved expressivity.
    """
    
    def __init__(self, n_angle_features: int, n_amplitude_features_log: int,
                 n_layers: int = 2, entanglement_strategy: str = "linear",
                 seed: Optional[int] = None):
        """
        Initialize the hybrid encoding classifier.
        
        Args:
            n_angle_features: Number of features for angle encoding
            n_amplitude_features_log: Log2 of number of amplitude features
            n_layers: Number of variational layers
            entanglement_strategy: Entanglement strategy
            seed: Random seed for weight initialization
        """
        self.n_angle_features = n_angle_features
        self.n_amplitude_features_log = n_amplitude_features_log
        self.n_amplitude_features = 2 ** n_amplitude_features_log
        self.n_total_features = n_angle_features + self.n_amplitude_features
        self.n_qubits = n_angle_features + n_amplitude_features_log
        self.n_layers = n_layers
        self.entanglement_strategy = entanglement_strategy
        
        self.circuit = build_hybrid_classifier(n_angle_features, n_amplitude_features_log,
                                              n_layers, entanglement_strategy)
        self.weights = initialize_hybrid_weights(self.n_qubits, n_layers, seed)
        self.complexity = benchmark_hybrid_circuit(n_angle_features, n_amplitude_features_log,
                                                  n_layers, entanglement_strategy)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid quantum circuit.
        
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
    
    def get_circuit_info(self) -> Dict[str, Union[int, str, Dict[str, int]]]:
        """
        Get comprehensive information about the hybrid circuit.
        
        Returns:
            Dictionary containing circuit specifications and complexity metrics
        """
        return {
            "n_angle_features": self.n_angle_features,
            "n_amplitude_features": self.n_amplitude_features,
            "n_total_features": self.n_total_features,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.weights.numel(),
            "encoding_type": "Hybrid (Angle + Amplitude)",
            "entanglement_strategy": self.entanglement_strategy,
            "variational_gates": "RX, RY, RZ rotations",
            "readout": "PauliZ on last qubit",
            "complexity_metrics": self.complexity
        }
    
    def get_complexity_summary(self) -> str:
        """
        Get a human-readable complexity summary.
        
        Returns:
            Formatted string with complexity information
        """
        c = self.complexity
        return f"""Hybrid Circuit Complexity:
        Total Qubits: {c['total_qubits']}
        Total Gates: {c['total_gates']}
        Circuit Depth: {c['circuit_depth']}
        
        Gate Breakdown:
        - Angle Encoding: {c['angle_encoding_gates']} gates
        - Amplitude Encoding: {c['amplitude_encoding_gates']} gates  
        - Cross Entanglement: {c['cross_entanglement_gates']} gates
        - Variational: {c['variational_gates']} gates
        - Layer Entanglement: {c['entanglement_gates']} gates
        """