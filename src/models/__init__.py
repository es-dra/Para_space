from typing import Dict, Optional

import torch.nn as nn

from .liif import LIIFDecoder, LIIFModel, FeatureEncoder
from .lte import LTEDecoder, LTEModel

MODEL_REGISTRY = {
    "liif": LIIFModel,
    "lte": LTEModel,
}

DECODER_REGISTRY = {
    "liif": LIIFDecoder,
    "lte": LTEDecoder,
}


def create_model(model_type: str, **kwargs) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


def create_decoder(model_type: str, **kwargs) -> nn.Module:
    if model_type not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder type: {model_type}. Available: {list(DECODER_REGISTRY.keys())}"
        )
    return DECODER_REGISTRY[model_type](**kwargs)


__all__ = [
    "LIIFDecoder",
    "LIIFModel",
    "LTEDecoder",
    "LTEModel",
    "FeatureEncoder",
    "MODEL_REGISTRY",
    "DECODER_REGISTRY",
    "create_model",
    "create_decoder",
]
