"""Postprocessing utilities."""


def stitch_chunks(chunk_transcripts: list[str]) -> str:
    """Join chunk transcripts into a final string."""
    cleaned = [part.strip() for part in chunk_transcripts if part and part.strip()]
    return " ".join(cleaned).strip()
