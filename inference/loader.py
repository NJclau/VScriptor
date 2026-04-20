"""Lazy model loading for ASR inference."""

import threading

import nemo.collections.asr as nemo_asr
import torch

from config import FALLBACK_MODEL_NAME

_model_lock = threading.Lock()
_model = None


def get_asr_model():
    """Return a lazily loaded singleton ASR model moved to the active device."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                    model_name=FALLBACK_MODEL_NAME,
                )
                _model = model.to(device.type)
    return _model
