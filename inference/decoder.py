"""CTC greedy decode from logits to text. Stateless."""

import torch
from transformers import Wav2Vec2BertProcessor

def decode_logits(logits: torch.Tensor, processor: Wav2Vec2BertProcessor) -> str:
    """Greedy decode: argmax over vocab, then batch_decode."""
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0] if transcription else ""
