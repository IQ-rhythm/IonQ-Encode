"""
Quantum Kitchen Sinks (QKS) implementation for quantum machine learning.

This module implements quantum kitchen sinks with random quantum features,
providing an efficient approximation to quantum kernels through randomized
quantum circuits with fixed parameters.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Callable
import math


def generate_random_parameters(n_qubits: int, n_layers: int, n_features: int,
                              parameter_type: str = "uniform", 
                              seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Generate random parameters for QKS circuits.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        n_features: Number of input features
        parameter_type: Type of parameter distribution ("uniform", "normal")
        seed: Random seed
    
    Returns:
        Dictionary of random parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    params = {}
    
    if parameter_type == "uniform":
        # Random parameters in [0, 2π]
        params["rotation_params"] = torch.rand(n_layers, n_qubits, 3) * 2 * np.pi
        params["data_params"] = torch.rand(n_features, n_qubits) * 2 * np.pi
        params["entanglement_params"] = torch.rand(n_layers, n_qubits) * 2 * np.pi
    elif parameter_type == "normal":
        # Random parameters from normal distribution
        params["rotation_params"] = torch.randn(n_layers, n_qubits, 3) * np.pi / 2
        params["data_params"] = torch.randn(n_features, n_qubits) * np.pi / 2
        params["entanglement_params"] = torch.randn(n_layers, n_qubits) * np.pi / 2
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")
    
    return params


def build_qks_circuit(n_qubits: int, n_layers: int, n_features: int,
                     entanglement_pattern: str = "linear",
                     random_params: Optional[Dict[str, torch.Tensor]] = None,
                     seed: Optional[int] = None) -> Tuple[qml.QNode, Dict[str, torch.Tensor]]:
    """
    Build a Quantum Kitchen Sinks circuit with random parameters.
    
    The QKS circuit structure:
    1. Data encoding with random linear combinations
    2. Random rotation layers (RX, RY, RZ with fixed random parameters)
    3. Random entanglement patterns
    4. Multiple layers for complexity
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of random layers
        n_features: Number of input features
        entanglement_pattern: Pattern for entanglement ("linear", "random", "all_to_all")
        random_params: Pre-generated random parameters (if None, generates new ones)
        seed: Random seed
    
    Returns:
        Tuple of (QKS circuit, random parameters used)
    """
    if random_params is None:
        random_params = generate_random_parameters(n_qubits, n_layers, n_features, seed=seed)
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def qks_circuit(features: torch.Tensor) -> List[float]:
        """
        Quantum Kitchen Sinks circuit.
        
        Args:
            features: Input features of shape (n_features,)
        
        Returns:
            List of expectation values from all qubits
        """
        # Pad or truncate features to match expected size
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            padding = torch.zeros(n_features - len(features))
            features = torch.cat([features, padding])
        
        # Step 1: Data encoding with random linear combinations
        apply_random_data_encoding(features, random_params["data_params"], n_qubits)
        
        # Step 2: Random quantum layers
        for layer in range(n_layers):
            # Random rotation layer
            apply_random_rotation_layer(random_params["rotation_params"][layer], n_qubits)
            
            # Random entanglement
            apply_random_entanglement(random_params["entanglement_params"][layer], 
                                    n_qubits, entanglement_pattern, layer)
        
        # Return expectation values from all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return qks_circuit, random_params


def apply_random_data_encoding(features: torch.Tensor, data_params: torch.Tensor, 
                              n_qubits: int) -> None:
    """
    Apply random data encoding to the quantum circuit.
    
    Args:
        features: Input features
        data_params: Random parameters for data encoding
        n_qubits: Number of qubits
    """
    # Random linear combinations of features encoded as rotations
    for qubit in range(n_qubits):
        # Compute random linear combination
        angle = 0.0
        for feat_idx in range(len(features)):
            angle += features[feat_idx] * data_params[feat_idx, qubit]
        
        # Apply as RY rotation (can be changed to RZ or RX)
        qml.RY(angle, wires=qubit)


def apply_random_rotation_layer(layer_params: torch.Tensor, n_qubits: int) -> None:
    """
    Apply a layer of random rotations.
    
    Args:
        layer_params: Random parameters for rotations of shape (n_qubits, 3)
        n_qubits: Number of qubits
    """
    for qubit in range(n_qubits):
        qml.RX(layer_params[qubit, 0], wires=qubit)
        qml.RY(layer_params[qubit, 1], wires=qubit)
        qml.RZ(layer_params[qubit, 2], wires=qubit)


def apply_random_entanglement(entangle_params: torch.Tensor, n_qubits: int,
                             pattern: str = "linear", layer_idx: int = 0) -> None:
    """
    Apply random entanglement pattern.
    
    Args:
        entangle_params: Random parameters for entanglement
        n_qubits: Number of qubits
        pattern: Entanglement pattern
        layer_idx: Layer index (for pattern variation)
    """
    if pattern == "linear":
        # Linear chain with random phases
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(entangle_params[i], wires=i + 1)
    
    elif pattern == "random":
        # Random connections based on layer and parameters
        np.random.seed(int(torch.sum(entangle_params).item()) + layer_idx)
        connections = []
        n_connections = min(n_qubits, max(1, n_qubits // 2))
        
        for _ in range(n_connections):
            i, j = np.random.choice(n_qubits, 2, replace=False)
            if (i, j) not in connections and (j, i) not in connections:
                connections.append((i, j))
                qml.CNOT(wires=[i, j])
                param_idx = min(i, len(entangle_params) - 1)
                qml.RZ(entangle_params[param_idx], wires=j)
    
    elif pattern == "all_to_all":
        # All-to-all (expensive but very expressive)
        param_idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if param_idx < len(entangle_params):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(entangle_params[param_idx % len(entangle_params)], wires=j)
                    param_idx += 1


def compute_qks_features(qks_circuit: qml.QNode, X: torch.Tensor) -> torch.Tensor:
    """
    Compute QKS features for a dataset.
    
    Args:
        qks_circuit: QKS quantum circuit
        X: Input data of shape (n_samples, n_features)
    
    Returns:
        QKS features of shape (n_samples, n_qubits)
    """
    n_samples = X.shape[0]
    
    # Compute features for all samples
    qks_features = []
    for i in range(n_samples):
        features = torch.tensor(qks_circuit(X[i]))
        qks_features.append(features)
    
    return torch.stack(qks_features)


def compute_qks_kernel_approximation(qks_circuit: qml.QNode, X: torch.Tensor, 
                                   Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute approximate kernel matrix using QKS features.
    
    Args:
        qks_circuit: QKS quantum circuit
        X: Training data
        Y: Test data (if None, uses X)
    
    Returns:
        Approximate kernel matrix
    """
    if Y is None:
        Y = X
    
    # Compute QKS features
    X_features = compute_qks_features(qks_circuit, X)
    Y_features = compute_qks_features(qks_circuit, Y)
    
    # Kernel matrix as inner products of features
    kernel_matrix = torch.mm(X_features, Y_features.T)
    
    return kernel_matrix


def benchmark_qks_complexity(n_qubits: int, n_layers: int, n_features: int,
                            entanglement_pattern: str = "linear") -> Dict[str, int]:
    """
    Benchmark the complexity of QKS circuits.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        n_features: Number of features
        entanglement_pattern: Entanglement pattern
    
    Returns:
        Complexity metrics
    """
    complexity = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_features": n_features,
        "total_gates": 0,
        "rotation_gates": 0,
        "cnot_gates": 0,
        "circuit_depth": 0
    }
    
    # Data encoding: RY gates for each qubit
    complexity["rotation_gates"] += n_qubits
    
    # Per layer: 3 rotations per qubit + entanglement
    rotation_per_layer = n_qubits * 3
    complexity["rotation_gates"] += rotation_per_layer * n_layers
    
    # Entanglement gates per layer
    cnot_per_layer = 0
    if entanglement_pattern == "linear":
        cnot_per_layer = n_qubits - 1
    elif entanglement_pattern == "random":
        cnot_per_layer = max(1, n_qubits // 2)
    elif entanglement_pattern == "all_to_all":
        cnot_per_layer = n_qubits * (n_qubits - 1) // 2
    
    complexity["cnot_gates"] = cnot_per_layer * n_layers
    
    # Total gates
    complexity["total_gates"] = complexity["rotation_gates"] + complexity["cnot_gates"]
    
    # Circuit depth estimate
    depth_per_layer = 3 + 2  # 3 rotations + entanglement depth
    complexity["circuit_depth"] = 1 + depth_per_layer * n_layers
    
    return complexity


class QuantumKitchenSinks:
    """
    Quantum Kitchen Sinks implementation for efficient quantum feature approximation.
    
    QKS provides an efficient way to approximate quantum kernels using
    randomized quantum circuits with fixed parameters.
    """
    
    def __init__(self, n_qubits: int, n_layers: int, n_features: int,
                 entanglement_pattern: str = "linear", parameter_type: str = "uniform",
                 seed: Optional[int] = None):
        """
        Initialize Quantum Kitchen Sinks.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of random layers
            n_features: Number of input features
            entanglement_pattern: Entanglement pattern
            parameter_type: Random parameter distribution type
            seed: Random seed
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.entanglement_pattern = entanglement_pattern
        self.parameter_type = parameter_type
        self.seed = seed
        
        # Generate random parameters
        self.random_params = generate_random_parameters(
            n_qubits, n_layers, n_features, parameter_type, seed
        )
        
        # Build QKS circuit
        self.circuit, _ = build_qks_circuit(
            n_qubits, n_layers, n_features, entanglement_pattern, 
            self.random_params, seed
        )
        
        # Compute complexity
        self.complexity = benchmark_qks_complexity(
            n_qubits, n_layers, n_features, entanglement_pattern
        )
    
    def __call__(self, features: torch.Tensor) -> List[float]:
        """Make QKS callable."""
        return self.circuit(features)
    
    def compute_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute QKS features for input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            QKS features of shape (n_samples, n_qubits)
        """
        return compute_qks_features(self.circuit, X)
    
    def compute_kernel_approximation(self, X: torch.Tensor, 
                                   Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute approximate kernel matrix.
        
        Args:
            X: Training data
            Y: Test data (if None, uses X)
        
        Returns:
            Approximate kernel matrix
        """
        return compute_qks_kernel_approximation(self.circuit, X, Y)
    
    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get QKS feature vector for a single sample.
        
        Args:
            x: Input sample
        
        Returns:
            QKS feature vector
        """
        return torch.tensor(self.circuit(x))
    
    def get_info(self) -> Dict[str, Union[str, int, Dict[str, int]]]:
        """Get QKS information."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_features": self.n_features,
            "entanglement_pattern": self.entanglement_pattern,
            "parameter_type": self.parameter_type,
            "complexity_metrics": self.complexity
        }
    
    def get_complexity_summary(self) -> str:
        """Get human-readable complexity summary."""
        c = self.complexity
        return f"""Quantum Kitchen Sinks Complexity:
        Qubits: {c['n_qubits']}
        Layers: {c['n_layers']}
        Features: {c['n_features']}
        Total Gates: {c['total_gates']}
        Circuit Depth: {c['circuit_depth']}
        
        Gate Breakdown:
        - Rotation Gates: {c['rotation_gates']}
        - CNOT Gates: {c['cnot_gates']}
        
        Pattern: {self.entanglement_pattern}
        Parameters: {self.parameter_type}
        """


def build_qks_classifier(n_qubits: int, n_layers: int, n_features: int,
                        entanglement_pattern: str = "linear") -> qml.QNode:
    """
    Build a QKS circuit optimized for binary classification.
    
    This differs from the general QKS circuit by:
    1. Using learnable variational parameters alongside fixed random parameters
    2. Returning a single logit from the last qubit
    3. Supporting gradient-based optimization
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        n_features: Number of input features
        entanglement_pattern: Entanglement pattern
    
    Returns:
        QKS classification circuit
    """
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def qks_classifier_circuit(features: torch.Tensor, weights: torch.Tensor, 
                              random_params: Dict[str, torch.Tensor]) -> float:
        """
        QKS classification circuit.
        
        Args:
            features: Input features
            weights: Learnable variational parameters
            random_params: Fixed random parameters
        
        Returns:
            Classification logit from Z expectation of last qubit
        """
        # Pad or truncate features
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            padding = torch.zeros(n_features - len(features))
            features = torch.cat([features, padding])
        
        # Step 1: Random data encoding
        apply_random_data_encoding(features, random_params["data_params"], n_qubits)
        
        # Step 2: Mixed random + variational layers
        for layer in range(n_layers):
            # Fixed random rotations
            apply_random_rotation_layer(random_params["rotation_params"][layer], n_qubits)
            
            # Learnable variational layer
            apply_qks_variational_layer(weights[layer], n_qubits)
            
            # Random entanglement
            apply_random_entanglement(random_params["entanglement_params"][layer], 
                                    n_qubits, entanglement_pattern, layer)
        
        # Return logit from Z expectation of last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return qks_classifier_circuit


def apply_qks_variational_layer(layer_weights: torch.Tensor, n_qubits: int) -> None:
    """
    Apply learnable variational layer for QKS classifier.
    
    Args:
        layer_weights: Variational parameters of shape (n_qubits, 2)
        n_qubits: Number of qubits
    """
    for i in range(n_qubits):
        qml.RY(layer_weights[i, 0], wires=i)  # Learnable amplitude rotation
        qml.RZ(layer_weights[i, 1], wires=i)  # Learnable phase rotation


def get_qks_weights_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    """
    Get the required shape for QKS classifier weights.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
    
    Returns:
        Weight shape (n_layers, n_qubits, 2)
    """
    return (n_layers, n_qubits, 2)


def initialize_qks_weights(n_qubits: int, n_layers: int, 
                          seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize learnable weights for QKS classifier.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        seed: Random seed
    
    Returns:
        Initialized weights
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    shape = get_qks_weights_shape(n_qubits, n_layers)
    weights = torch.randn(shape) * (np.pi / 4)
    weights.requires_grad_(True)
    
    return weights


def create_qks_model(n_features: int, n_layers: int = 2, 
                    entanglement_pattern: str = "linear") -> Tuple[qml.QNode, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Create a complete QKS classification model.
    
    Args:
        n_features: Number of input features
        n_layers: Number of layers
        entanglement_pattern: Entanglement pattern
    
    Returns:
        Tuple of (circuit, weights, random_params)
    """
    n_qubits = n_features
    circuit = build_qks_classifier(n_qubits, n_layers, n_features, entanglement_pattern)
    weights = initialize_qks_weights(n_qubits, n_layers)
    random_params = generate_random_parameters(n_qubits, n_layers, n_features)
    
    return circuit, weights, random_params


class QKSClassifier:
    """
    A PyTorch-compatible wrapper for QKS quantum classifier.
    
    This class provides a scikit-learn-like interface for QKS circuits
    with both fixed random parameters and learnable variational parameters.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of QKS layers.
        circuit (qml.QNode): The quantum circuit function.
        weights (torch.Tensor): Trainable variational parameters.
        random_params (Dict[str, torch.Tensor]): Fixed random parameters.
    """
    
    def __init__(self, n_features: int, n_layers: int = 2, 
                 entanglement_pattern: str = "linear", seed: Optional[int] = None):
        """
        Initialize the QKS classifier.
        
        Args:
            n_features: Number of input features
            n_layers: Number of QKS layers
            entanglement_pattern: Entanglement pattern
            seed: Random seed for initialization
        """
        self.n_qubits = n_features
        self.n_layers = n_layers
        self.entanglement_pattern = entanglement_pattern
        
        self.circuit = build_qks_classifier(n_features, n_layers, n_features, entanglement_pattern)
        self.weights = initialize_qks_weights(n_features, n_layers, seed)
        self.random_params = generate_random_parameters(n_features, n_layers, n_features, seed=seed)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QKS circuit.
        
        Args:
            features: Input features of shape (batch_size, n_features) or (n_features,)
        
        Returns:
            Predictions of shape (batch_size,) or scalar for single samples
        """
        if len(features.shape) == 1:
            # Single sample
            return self.circuit(features, self.weights, self.random_params)
        else:
            # Batch processing
            batch_size = features.shape[0]
            predictions = torch.zeros(batch_size)
            
            for i in range(batch_size):
                predictions[i] = self.circuit(features[i], self.weights, self.random_params)
            
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
        complexity = benchmark_qks_complexity(
            self.n_qubits, self.n_layers, self.n_qubits, self.entanglement_pattern
        )
        
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.weights.numel(),
            "encoding_type": "Quantum Kitchen Sinks (random features)",
            "entanglement": f"{self.entanglement_pattern} pattern",
            "variational_gates": "RY, RZ rotations",
            "complexity_metrics": complexity
        }


class EnsembleQKS:
    """
    Ensemble of Quantum Kitchen Sinks for improved approximation.
    
    Uses multiple QKS with different random parameters to create
    a richer feature representation.
    """
    
    def __init__(self, n_qubits: int, n_layers: int, n_features: int,
                 n_estimators: int = 10, entanglement_pattern: str = "linear",
                 parameter_type: str = "uniform", base_seed: Optional[int] = None):
        """
        Initialize ensemble of QKS.
        
        Args:
            n_qubits: Number of qubits per QKS
            n_layers: Number of layers per QKS
            n_features: Number of input features
            n_estimators: Number of QKS in ensemble
            entanglement_pattern: Entanglement pattern
            parameter_type: Parameter distribution type
            base_seed: Base seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.entanglement_pattern = entanglement_pattern
        self.parameter_type = parameter_type
        self.base_seed = base_seed
        
        # Create ensemble of QKS
        self.qks_ensemble = []
        for i in range(n_estimators):
            seed = base_seed + i if base_seed is not None else None
            qks = QuantumKitchenSinks(
                n_qubits, n_layers, n_features, entanglement_pattern,
                parameter_type, seed
            )
            self.qks_ensemble.append(qks)
    
    def compute_ensemble_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble features by concatenating all QKS features.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Ensemble features of shape (n_samples, n_estimators * n_qubits)
        """
        all_features = []
        for qks in self.qks_ensemble:
            features = qks.compute_features(X)
            all_features.append(features)
        
        return torch.cat(all_features, dim=1)
    
    def compute_ensemble_kernel(self, X: torch.Tensor, 
                              Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ensemble kernel as average of individual QKS kernels.
        
        Args:
            X: Training data
            Y: Test data (if None, uses X)
        
        Returns:
            Ensemble kernel matrix
        """
        ensemble_kernel = None
        
        for qks in self.qks_ensemble:
            kernel = qks.compute_kernel_approximation(X, Y)
            if ensemble_kernel is None:
                ensemble_kernel = kernel
            else:
                ensemble_kernel += kernel
        
        return ensemble_kernel / self.n_estimators
    
    def get_info(self) -> Dict[str, Union[str, int]]:
        """Get ensemble information."""
        single_complexity = self.qks_ensemble[0].complexity
        return {
            "n_estimators": self.n_estimators,
            "n_qubits_per_estimator": self.n_qubits,
            "total_qubits": self.n_qubits * self.n_estimators,
            "n_layers": self.n_layers,
            "n_features": self.n_features,
            "total_gates_per_estimator": single_complexity["total_gates"],
            "total_gates_ensemble": single_complexity["total_gates"] * self.n_estimators,
            "entanglement_pattern": self.entanglement_pattern,
            "parameter_type": self.parameter_type
        }