"""Lazy HF model loading for ASR inference."""

import threading

import torch
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from config import MODEL_NAME

_model_lock = threading.Lock()
_model = None
_processor = None


def get_model_and_processor() -> tuple[Wav2Vec2BertForCTC, Wav2Vec2BertProcessor]:
    """Return lazily loaded singleton model and processor on the active device."""
    global _model, _processor

    if _model is None or _processor is None:
        with _model_lock:
            if _model is None or _processor is None:
                _processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_NAME)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _model = Wav2Vec2BertForCTC.from_pretrained(MODEL_NAME).to(device)
                _model.eval()

    return _model, _processor
