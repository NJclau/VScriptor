"""Lazy model and processor loader for Hugging Face Wav2Vec2-BERT CTC."""

from threading import Lock

from config import MODEL_NAME

_model = None
_processor = None
_loader_lock = Lock()


def get_model_and_processor():
    """Return a singleton (model, processor) pair loaded lazily on first call."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    with _loader_lock:
        if _model is None or _processor is None:
            from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

            _processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_NAME)
            _model = Wav2Vec2BertForCTC.from_pretrained(MODEL_NAME)

    return _model, _processor
