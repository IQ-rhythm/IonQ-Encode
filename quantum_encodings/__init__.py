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

__all__ = [
    "build_ae_classifier",
    "encode_features", 
    "apply_entangling_layer",
    "apply_variational_layer",
    "get_ae_weights_shape",
    "initialize_ae_weights",
    "create_ae_model",
    "AngleEncodingClassifier"
]