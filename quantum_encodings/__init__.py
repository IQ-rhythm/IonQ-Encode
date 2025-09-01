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
    "AmplitudeEncodingClassifier"
]