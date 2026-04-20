"""Lazy model loading for ASR inference."""

import threading

from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from config import MODEL_NAME

_model_lock = threading.Lock()
_model = None
_processor = None


def get_model_and_processor():
    """Return lazily loaded singleton model and processor."""
    global _model, _processor
    if _model is None or _processor is None:
        with _model_lock:
            if _model is None:
                _model = Wav2Vec2BertForCTC.from_pretrained(MODEL_NAME)
                _processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor
