"""
Quantum encoding methods for machine learning.

This package contains implementations of various quantum data encoding strategies
for use with quantum machine learning algorithms on near-term quantum devices.
"""

from .angle_encoding import (
    build_ae_classifier,
    encode_features,
    apply_entangling_layer,
    apply_variational_layer,
    get_ae_weights_shape,
    initialize_ae_weights,
    create_ae_model,
    AngleEncodingClassifier
)

from .amplitude_encoding import (
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

from .hybrid_encoding import (
    split_features_for_hybrid,
    validate_hybrid_encoding_params,
    build_hybrid_classifier,
    get_hybrid_weights_shape,
    initialize_hybrid_weights,
    benchmark_hybrid_circuit,
    create_hybrid_model,
    HybridEncodingClassifier
)

from .kernel_feature_map import (
    validate_feature_map_params,
    build_zz_feature_map,
    build_iqp_feature_map,
    compute_kernel_matrix,
    get_feature_map_complexity,
    QuantumKernelFeatureMap,
    create_feature_map_model,
    build_kernel_classifier,
    get_kernel_classifier_weights_shape,
    initialize_kernel_classifier_weights,
    create_kernel_classifier_model,
    KernelFeatureMapClassifier
)

from .data_reuploading import (
    build_dru_classifier,
    initialize_dru_weights,
    initialize_dru_weights_torch,
    get_dru_weights_shape,
    create_dru_model,
    dru_feature_dimension,
    DRUClassifier
)

from .qks import (
    generate_random_parameters,
    build_qks_circuit,
    compute_qks_features,
    compute_qks_kernel_approximation,
    benchmark_qks_complexity,
    QuantumKitchenSinks,
    EnsembleQKS,
    build_qks_classifier,
    get_qks_weights_shape,
    initialize_qks_weights,
    create_qks_model,
    QKSClassifier
)

__all__ = [
    "build_ae_classifier",
    "encode_features", 
    "apply_entangling_layer",
    "apply_variational_layer",
    "get_ae_weights_shape",
    "initialize_ae_weights",
    "create_ae_model",
    "AngleEncodingClassifier",
    "normalize_vector",
    "pad_or_truncate_vector",
    "validate_amplitude_encoding_size",
    "build_exact_amplitude_classifier",
    "build_approximate_amplitude_classifier",
    "get_amplitude_weights_shape",
    "initialize_amplitude_weights",
    "create_amplitude_model",
    "AmplitudeEncodingClassifier",
    "split_features_for_hybrid",
    "validate_hybrid_encoding_params",
    "build_hybrid_classifier",
    "get_hybrid_weights_shape",
    "initialize_hybrid_weights",
    "benchmark_hybrid_circuit",
    "create_hybrid_model",
    "HybridEncodingClassifier",
    "validate_feature_map_params",
    "build_zz_feature_map",
    "build_iqp_feature_map",
    "compute_kernel_matrix",
    "get_feature_map_complexity",
    "QuantumKernelFeatureMap",
    "create_feature_map_model",
    "build_dru_classifier",
    "initialize_dru_weights",
    "initialize_dru_weights_torch",
    "get_dru_weights_shape",
    "create_dru_model",
    "dru_feature_dimension",
    "DRUClassifier",
    "generate_random_parameters",
    "build_qks_circuit",
    "compute_qks_features",
    "compute_qks_kernel_approximation",
    "benchmark_qks_complexity",
    "QuantumKitchenSinks",
    "EnsembleQKS",
    "build_qks_classifier",
    "get_qks_weights_shape",
    "initialize_qks_weights",
    "create_qks_model",
    "QKSClassifier",
    "build_kernel_classifier",
    "get_kernel_classifier_weights_shape",
    "initialize_kernel_classifier_weights",
    "create_kernel_classifier_model",
    "KernelFeatureMapClassifier"
]