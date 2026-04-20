"""CTC decoding helpers for HF logits."""

import torch


def decode_logits(logits: torch.Tensor, processor) -> str:
    """Decode model logits with greedy CTC decoding via processor.batch_decode."""
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded_batch = processor.batch_decode(predicted_ids)
    if not decoded_batch:
        return ""
    return decoded_batch[0].strip()
