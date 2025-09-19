"""
Kernel Feature Maps implementation for quantum machine learning.

This module implements quantum kernel feature maps including ZZFeatureMap
and IQPFeatureMap (Instantaneous Quantum Polynomial) for quantum kernel
methods and support vector machines.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Callable
import math
import itertools


def validate_feature_map_params(n_qubits: int, n_features: int, repetitions: int = 1) -> None:
    """
    Validate feature map parameters.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of input features
        repetitions: Number of repetitions of the feature map
    
    Raises:
        ValueError: If parameters are invalid
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if n_features > n_qubits:
        raise ValueError("n_features cannot exceed n_qubits")


def build_zz_feature_map(n_qubits: int, n_features: int, repetitions: int = 2,
                        entanglement: str = "linear") -> qml.QNode:
    """
    Build a ZZFeatureMap quantum circuit for kernel methods.
    
    The ZZFeatureMap implements:
    1. Hadamard gates on all qubits for superposition
    2. RZ rotations with feature values (first-order terms)
    3. ZZ interactions between entangled qubits (second-order terms)
    4. Repeat the encoding for specified repetitions
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_features: Number of input features (≤ n_qubits)
        repetitions: Number of repetitions of the feature map
        entanglement: Entanglement pattern ("linear", "circular", "full")
    
    Returns:
        Quantum circuit function for ZZ feature mapping
    """
    validate_feature_map_params(n_qubits, n_features, repetitions)
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def zz_feature_circuit(features: torch.Tensor) -> List[float]:
        """
        ZZFeatureMap quantum circuit.
        
        Args:
            features: Input features of shape (n_features,)
        
        Returns:
            List of expectation values from all qubits
        """
        # Ensure features have correct length
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            # Pad with zeros
            padding = torch.zeros(n_features - len(features))
            features = torch.cat([features, padding])
        
        # Apply the ZZ feature map with repetitions
        for rep in range(repetitions):
            # Step 1: Hadamard gates for superposition (first repetition only)
            if rep == 0:
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
            
            # Step 2: First-order feature encoding with RZ rotations
            for i in range(n_features):
                qml.RZ(features[i] * 2.0, wires=i)  # Scale features for better discrimination
            
            # Step 3: Second-order ZZ interactions based on entanglement pattern
            apply_zz_interactions(features, n_qubits, n_features, entanglement)
        
        # Return expectation values from all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return zz_feature_circuit


def apply_zz_interactions(features: torch.Tensor, n_qubits: int, n_features: int,
                         entanglement: str = "linear") -> None:
    """
    Apply ZZ interactions between qubits based on entanglement pattern.
    
    Args:
        features: Input features
        n_qubits: Number of qubits
        n_features: Number of features
        entanglement: Entanglement pattern
    """
    if entanglement == "linear":
        # Linear chain: ZZ between adjacent qubits
        for i in range(min(n_qubits - 1, n_features - 1)):
            # ZZ interaction: exp(i * φ * Z_i ⊗ Z_{i+1})
            # Implemented as: CNOT(i, i+1), RZ(φ, i+1), CNOT(i, i+1)
            phi = features[i] * features[i + 1] if i + 1 < len(features) else features[i] * features[0]
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(phi, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
    
    elif entanglement == "circular":
        # Circular: linear + wraparound connection
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            phi = features[i % n_features] * features[j % n_features]
            qml.CNOT(wires=[i, j])
            qml.RZ(phi, wires=j)
            qml.CNOT(wires=[i, j])
    
    elif entanglement == "full":
        # Full: all pairs of qubits (expensive but expressive)
        for i in range(min(n_qubits, n_features)):
            for j in range(i + 1, min(n_qubits, n_features)):
                phi = features[i] * features[j]
                qml.CNOT(wires=[i, j])
                qml.RZ(phi, wires=j)
                qml.CNOT(wires=[i, j])


def build_iqp_feature_map(n_qubits: int, n_features: int, repetitions: int = 1) -> qml.QNode:
    """
    Build an IQP (Instantaneous Quantum Polynomial) feature map.
    
    The IQPFeatureMap implements:
    1. Hadamard gates on all qubits
    2. RZ rotations with feature values (diagonal unitaries)
    3. All-to-all ZZ interactions (creates polynomial kernel)
    4. Repeat for specified repetitions
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_features: Number of input features (≤ n_qubits)
        repetitions: Number of repetitions
    
    Returns:
        Quantum circuit function for IQP feature mapping
    """
    validate_feature_map_params(n_qubits, n_features, repetitions)
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def iqp_feature_circuit(features: torch.Tensor) -> List[float]:
        """
        IQP feature map quantum circuit.
        
        Args:
            features: Input features of shape (n_features,)
        
        Returns:
            List of expectation values from all qubits
        """
        # Ensure features have correct length
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            padding = torch.zeros(n_features - len(features))
            features = torch.cat([features, padding])
        
        # Apply the IQP feature map with repetitions
        for rep in range(repetitions):
            # Step 1: Hadamard gates for superposition (first repetition only)
            if rep == 0:
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
            
            # Step 2: First-order terms (diagonal unitaries)
            for i in range(n_features):
                qml.RZ(features[i] * 2.0, wires=i)  # Scale features for better discrimination
            
            # Step 3: Second-order terms (all pairs)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if j < n_qubits:  # Ensure we don't exceed qubit count
                        phi = features[i] * features[j]
                        # ZZ interaction
                        qml.CNOT(wires=[i, j])
                        qml.RZ(phi, wires=j)
                        qml.CNOT(wires=[i, j])
        
        # Return expectation values from all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return iqp_feature_circuit


def compute_kernel_matrix(feature_map: qml.QNode, X: torch.Tensor, Y: Optional[torch.Tensor] = None,
                         normalize: bool = True) -> torch.Tensor:
    """
    Compute the quantum kernel matrix using a feature map.
    
    The kernel value between two samples x_i and x_j is computed as:
    K(x_i, x_j) = |⟨0|U†(x_i)U(x_j)|0⟩|²
    
    This is approximated using the inner product of expectation values.
    
    Args:
        feature_map: Quantum feature map circuit
        X: Training data of shape (n_samples, n_features)
        Y: Test data of shape (m_samples, n_features). If None, uses X (symmetric matrix)
        normalize: Whether to normalize kernel values
    
    Returns:
        Kernel matrix of shape (n_samples, m_samples)
    """
    if Y is None:
        Y = X
    
    n_samples = X.shape[0]
    m_samples = Y.shape[0]
    
    # Get feature map outputs for all samples
    X_features = []
    for i in range(n_samples):
        features = torch.tensor(feature_map(X[i]))
        X_features.append(features)
    
    Y_features = []
    for i in range(m_samples):
        if torch.equal(X, Y) and i < len(X_features):
            Y_features.append(X_features[i])  # Reuse computation for symmetric case
        else:
            features = torch.tensor(feature_map(Y[i]))
            Y_features.append(features)
    
    # Stack features into matrices for efficient computation
    X_feature_matrix = torch.stack(X_features)  # (n_samples, n_qubits)
    Y_feature_matrix = torch.stack(Y_features)  # (m_samples, n_qubits)
    
    # Compute kernel matrix as cosine similarity (normalized inner products)
    if normalize:
        # Normalize feature vectors
        X_norms = torch.norm(X_feature_matrix, dim=1, keepdim=True)
        Y_norms = torch.norm(Y_feature_matrix, dim=1, keepdim=True)
        
        # Avoid division by zero
        X_norms = torch.clamp(X_norms, min=1e-8)
        Y_norms = torch.clamp(Y_norms, min=1e-8)
        
        X_normalized = X_feature_matrix / X_norms
        Y_normalized = Y_feature_matrix / Y_norms
        
        kernel_matrix = torch.mm(X_normalized, Y_normalized.T)
    else:
        # Simple inner product kernel
        kernel_matrix = torch.mm(X_feature_matrix, Y_feature_matrix.T)
    
    # Transform to ensure positive semi-definiteness and reasonable range
    # Apply a polynomial kernel transformation: (γ * K + c)^d
    gamma = 1.0
    c = 0.1
    d = 1
    kernel_matrix = gamma * kernel_matrix + c
    
    # Ensure positive values while preserving differences
    kernel_matrix = torch.abs(kernel_matrix)
    
    return kernel_matrix


def get_feature_map_complexity(feature_map_type: str, n_qubits: int, n_features: int,
                              repetitions: int = 1, entanglement: str = "linear") -> Dict[str, int]:
    """
    Analyze the complexity of different feature maps.
    
    Args:
        feature_map_type: Type of feature map ("zz" or "iqp")
        n_qubits: Number of qubits
        n_features: Number of features
        repetitions: Number of repetitions
        entanglement: Entanglement pattern (for ZZ feature map)
    
    Returns:
        Dictionary with complexity metrics
    """
    complexity = {
        "n_qubits": n_qubits,
        "n_features": n_features,
        "repetitions": repetitions,
        "total_gates": 0,
        "hadamard_gates": 0,
        "rz_gates": 0,
        "cnot_gates": 0,
        "circuit_depth": 0
    }
    
    if feature_map_type.lower() == "zz":
        # ZZ Feature Map complexity
        complexity["hadamard_gates"] = n_qubits  # Applied once
        
        # Per repetition
        rz_per_rep = n_features  # First-order terms
        
        # Second-order ZZ interactions
        cnot_per_rep = 0
        if entanglement == "linear":
            cnot_per_rep = 2 * min(n_qubits - 1, n_features - 1)
            rz_per_rep += min(n_qubits - 1, n_features - 1)
        elif entanglement == "circular":
            cnot_per_rep = 2 * n_qubits
            rz_per_rep += n_qubits
        elif entanglement == "full":
            n_pairs = min(n_features * (n_features - 1) // 2, n_qubits * (n_qubits - 1) // 2)
            cnot_per_rep = 2 * n_pairs
            rz_per_rep += n_pairs
        
        complexity["rz_gates"] = rz_per_rep * repetitions
        complexity["cnot_gates"] = cnot_per_rep * repetitions
        
        # Rough depth estimate
        complexity["circuit_depth"] = 1 + repetitions * (2 + 3)  # H + per rep (RZ + ZZ interactions)
    
    elif feature_map_type.lower() == "iqp":
        # IQP Feature Map complexity
        complexity["hadamard_gates"] = n_qubits
        
        # Per repetition
        rz_first_order = n_features
        n_pairs = n_features * (n_features - 1) // 2
        rz_second_order = n_pairs
        cnot_interactions = 2 * n_pairs  # 2 CNOTs per ZZ interaction
        
        complexity["rz_gates"] = (rz_first_order + rz_second_order) * repetitions
        complexity["cnot_gates"] = cnot_interactions * repetitions
        
        # Depth estimate
        complexity["circuit_depth"] = 1 + repetitions * (1 + 3 * n_pairs)  # H + per rep (RZ + ZZ depth)
    
    complexity["total_gates"] = (complexity["hadamard_gates"] + 
                                complexity["rz_gates"] + 
                                complexity["cnot_gates"])
    
    return complexity


class QuantumKernelFeatureMap:
    """
    A unified interface for quantum kernel feature maps.
    
    This class provides a consistent interface for different quantum feature maps
    used in quantum kernel methods and quantum SVMs.
    """
    
    def __init__(self, feature_map_type: str, n_qubits: int, n_features: int,
                 repetitions: int = 1, entanglement: str = "linear",
                 seed: Optional[int] = None):
        """
        Initialize the quantum kernel feature map.
        
        Args:
            feature_map_type: Type of feature map ("zz", "iqp")
            n_qubits: Number of qubits
            n_features: Number of input features
            repetitions: Number of repetitions of the feature map
            entanglement: Entanglement pattern (for ZZ feature map)
            seed: Random seed (for reproducibility)
        """
        self.feature_map_type = feature_map_type.lower()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.repetitions = repetitions
        self.entanglement = entanglement
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Build the appropriate feature map
        if self.feature_map_type == "zz":
            self.circuit = build_zz_feature_map(n_qubits, n_features, repetitions, entanglement)
        elif self.feature_map_type == "iqp":
            self.circuit = build_iqp_feature_map(n_qubits, n_features, repetitions)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Compute complexity metrics
        self.complexity = get_feature_map_complexity(
            self.feature_map_type, n_qubits, n_features, repetitions, entanglement
        )
    
    def __call__(self, features: torch.Tensor) -> List[float]:
        """Make the feature map callable."""
        return self.circuit(features)
    
    def compute_kernel_matrix(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None,
                             normalize: bool = True) -> torch.Tensor:
        """
        Compute kernel matrix for the given data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            Y: Test data. If None, computes symmetric kernel matrix
            normalize: Whether to normalize kernel values
        
        Returns:
            Kernel matrix
        """
        return compute_kernel_matrix(self.circuit, X, Y, normalize)
    
    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the quantum feature vector for a single sample.
        
        Args:
            x: Input sample of shape (n_features,)
        
        Returns:
            Feature vector (expectation values)
        """
        return torch.tensor(self.circuit(x))
    
    def get_info(self) -> Dict[str, Union[str, int, Dict[str, int]]]:
        """
        Get information about the feature map.
        
        Returns:
            Dictionary with feature map specifications
        """
        return {
            "feature_map_type": self.feature_map_type,
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "repetitions": self.repetitions,
            "entanglement": self.entanglement,
            "complexity_metrics": self.complexity
        }
    
    def get_complexity_summary(self) -> str:
        """
        Get a human-readable complexity summary.
        
        Returns:
            Formatted string with complexity information
        """
        c = self.complexity
        return f"""Quantum Feature Map Complexity ({self.feature_map_type.upper()}):
        Qubits: {c['n_qubits']}
        Features: {c['n_features']} 
        Repetitions: {c['repetitions']}
        Total Gates: {c['total_gates']}
        Circuit Depth: {c['circuit_depth']}
        
        Gate Breakdown:
        - Hadamard: {c['hadamard_gates']} gates
        - RZ: {c['rz_gates']} gates
        - CNOT: {c['cnot_gates']} gates
        """


def create_feature_map_model(feature_map_type: str, n_qubits: int, n_features: int,
                            repetitions: int = 1, entanglement: str = "linear",
                            seed: Optional[int] = None) -> Tuple[QuantumKernelFeatureMap, Dict[str, int]]:
    """
    Create a complete quantum kernel feature map model.
    
    Args:
        feature_map_type: Type of feature map ("zz", "iqp")
        n_qubits: Number of qubits
        n_features: Number of input features
        repetitions: Number of repetitions
        entanglement: Entanglement pattern
        seed: Random seed
    
    Returns:
        Tuple of (feature_map, complexity_metrics)
    """
    feature_map = QuantumKernelFeatureMap(
        feature_map_type, n_qubits, n_features, repetitions, entanglement, seed
    )
    
    return feature_map, feature_map.complexity


def build_kernel_classifier(n_qubits: int, n_features: int, feature_map_type: str = "zz",
                           repetitions: int = 1, entanglement: str = "linear") -> qml.QNode:
    """
    Build a quantum kernel classifier adapted for QNN execution.
    
    This converts kernel feature maps into classifiers by:
    1. Using the feature map circuit for encoding
    2. Adding learnable variational parameters
    3. Returning a single logit from the last qubit
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of input features
        feature_map_type: Type of feature map ("zz", "iqp")
        repetitions: Number of repetitions
        entanglement: Entanglement pattern
    
    Returns:
        Quantum kernel classifier circuit
    """
    validate_feature_map_params(n_qubits, n_features, repetitions)
    
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device, interface="torch")
    def kernel_classifier_circuit(features: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Kernel classifier circuit.
        
        Args:
            features: Input features
            weights: Learnable variational parameters
        
        Returns:
            Classification logit from Z expectation of last qubit
        """
        # Ensure features have correct length
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            padding = torch.zeros(n_features - len(features))
            features = torch.cat([features, padding])
        
        # Apply kernel feature map encoding
        if feature_map_type.lower() == "zz":
            apply_zz_kernel_encoding(features, weights, n_qubits, n_features, repetitions, entanglement)
        elif feature_map_type.lower() == "iqp":
            apply_iqp_kernel_encoding(features, weights, n_qubits, n_features, repetitions)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Return classification logit from Z expectation of last qubit
        return qml.expval(qml.PauliZ(n_qubits - 1))
    
    return kernel_classifier_circuit


def apply_zz_kernel_encoding(features: torch.Tensor, weights: torch.Tensor,
                            n_qubits: int, n_features: int, repetitions: int,
                            entanglement: str = "linear") -> None:
    """
    Apply ZZ kernel encoding with learnable parameters.
    
    Args:
        features: Input features
        weights: Learnable parameters
        n_qubits: Number of qubits
        n_features: Number of features
        repetitions: Number of repetitions
        entanglement: Entanglement pattern
    """
    weight_idx = 0
    
    # Apply ZZ feature map with learnable parameters
    for rep in range(repetitions):
        # Step 1: Hadamard gates (first repetition only)
        if rep == 0:
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        
        # Step 2: First-order feature encoding with learnable scaling
        for i in range(n_features):
            # Learnable scaling of feature encoding
            scaling = weights[weight_idx] if weight_idx < len(weights) else 1.0
            qml.RZ(features[i] * scaling, wires=i)
            weight_idx += 1
        
        # Step 3: ZZ interactions with learnable parameters
        apply_zz_interactions(features, n_qubits, n_features, entanglement)
        
        # Step 4: Learnable variational layer
        for i in range(min(n_qubits, len(weights) - weight_idx)):
            qml.RY(weights[weight_idx], wires=i)
            weight_idx += 1
            if weight_idx < len(weights):
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1


def apply_iqp_kernel_encoding(features: torch.Tensor, weights: torch.Tensor,
                             n_qubits: int, n_features: int, repetitions: int) -> None:
    """
    Apply IQP kernel encoding with learnable parameters.
    
    Args:
        features: Input features
        weights: Learnable parameters
        n_qubits: Number of qubits
        n_features: Number of features
        repetitions: Number of repetitions
    """
    weight_idx = 0
    
    # Apply IQP feature map with learnable parameters
    for rep in range(repetitions):
        # Step 1: Hadamard gates (first repetition only)
        if rep == 0:
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
        
        # Step 2: First-order terms with learnable scaling
        for i in range(n_features):
            scaling = weights[weight_idx] if weight_idx < len(weights) else 1.0
            qml.RZ(features[i] * scaling, wires=i)
            weight_idx += 1
        
        # Step 3: Second-order terms (all pairs)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if j < n_qubits:
                    phi = features[i] * features[j]
                    # ZZ interaction
                    qml.CNOT(wires=[i, j])
                    qml.RZ(phi, wires=j)
                    qml.CNOT(wires=[i, j])
        
        # Step 4: Learnable variational layer
        for i in range(min(n_qubits, (len(weights) - weight_idx) // 2)):
            if weight_idx + 1 < len(weights):
                qml.RY(weights[weight_idx], wires=i)
                qml.RZ(weights[weight_idx + 1], wires=i)
                weight_idx += 2


def get_kernel_classifier_weights_shape(n_qubits: int, n_features: int, 
                                       feature_map_type: str = "zz",
                                       repetitions: int = 1) -> int:
    """
    Get the required number of weights for kernel classifier.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of features
        feature_map_type: Type of feature map
        repetitions: Number of repetitions
    
    Returns:
        Number of required weights
    """
    if feature_map_type.lower() == "zz":
        # Per repetition: n_features (scaling) + 2*n_qubits (variational RY+RZ)
        weights_per_rep = n_features + 2 * n_qubits
        return weights_per_rep * repetitions
    elif feature_map_type.lower() == "iqp":
        # Per repetition: n_features (scaling) + 2*n_qubits (variational RY+RZ)
        weights_per_rep = n_features + 2 * n_qubits
        return weights_per_rep * repetitions
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")


def initialize_kernel_classifier_weights(n_qubits: int, n_features: int,
                                        feature_map_type: str = "zz",
                                        repetitions: int = 1,
                                        seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize weights for kernel classifier.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of features
        feature_map_type: Type of feature map
        repetitions: Number of repetitions
        seed: Random seed
    
    Returns:
        Initialized weights
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_weights = get_kernel_classifier_weights_shape(n_qubits, n_features, feature_map_type, repetitions)
    weights = torch.randn(n_weights) * (np.pi / 4)
    weights.requires_grad_(True)
    
    return weights


def create_kernel_classifier_model(n_features: int, feature_map_type: str = "zz",
                                  repetitions: int = 1, entanglement: str = "linear") -> Tuple[qml.QNode, torch.Tensor]:
    """
    Create a complete kernel classifier model.
    
    Args:
        n_features: Number of input features
        feature_map_type: Type of feature map
        repetitions: Number of repetitions
        entanglement: Entanglement pattern
    
    Returns:
        Tuple of (circuit, weights)
    """
    n_qubits = n_features
    circuit = build_kernel_classifier(n_qubits, n_features, feature_map_type, repetitions, entanglement)
    weights = initialize_kernel_classifier_weights(n_qubits, n_features, feature_map_type, repetitions)
    
    return circuit, weights


class KernelFeatureMapClassifier:
    """
    A PyTorch-compatible wrapper for kernel feature map quantum classifier.
    
    This class adapts quantum kernel feature maps for supervised learning
    by adding learnable variational parameters for gradient-based optimization.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_features (int): Number of input features.
        feature_map_type (str): Type of feature map ("zz", "iqp").
        repetitions (int): Number of repetitions.
        entanglement (str): Entanglement pattern.
        circuit (qml.QNode): The quantum circuit function.
        weights (torch.Tensor): Trainable parameters.
    """
    
    def __init__(self, n_features: int, feature_map_type: str = "zz",
                 repetitions: int = 1, entanglement: str = "linear",
                 seed: Optional[int] = None):
        """
        Initialize the kernel feature map classifier.
        
        Args:
            n_features: Number of input features
            feature_map_type: Type of feature map ("zz", "iqp")
            repetitions: Number of repetitions
            entanglement: Entanglement pattern
            seed: Random seed for initialization
        """
        self.n_qubits = n_features
        self.n_features = n_features
        self.feature_map_type = feature_map_type.lower()
        self.repetitions = repetitions
        self.entanglement = entanglement
        
        self.circuit = build_kernel_classifier(n_features, n_features, feature_map_type, repetitions, entanglement)
        self.weights = initialize_kernel_classifier_weights(n_features, n_features, feature_map_type, repetitions, seed)
        
        # Compute complexity metrics
        self.complexity = get_feature_map_complexity(
            feature_map_type, n_features, n_features, repetitions, entanglement
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the kernel classifier circuit.
        
        Args:
            features: Input features of shape (batch_size, n_features) or (n_features,)
        
        Returns:
            Predictions of shape (batch_size,) or scalar for single samples
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
            "n_features": self.n_features,
            "n_parameters": len(self.weights),
            "feature_map_type": self.feature_map_type.upper(),
            "repetitions": self.repetitions,
            "encoding_type": f"Kernel Feature Map ({self.feature_map_type.upper()}) with variational parameters",
            "entanglement": f"{self.entanglement} pattern",
            "variational_gates": "RY, RZ rotations",
            "complexity_metrics": self.complexity
        }