"""Logits decoding utilities."""


def decode_logits_ctc_greedy(logits, processor) -> str:
    """Decode model logits using CTC greedy decoding and return text."""
    if hasattr(logits, "argmax"):
        try:
            predicted_ids = logits.argmax(dim=-1)
        except TypeError:
            predicted_ids = logits.argmax(axis=-1)
    else:
        raise ValueError("Unsupported logits type for greedy CTC decoding.")
    transcript = processor.batch_decode(predicted_ids)[0]
    return transcript.strip()
