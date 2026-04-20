"""Chunk decode helpers."""

import os


def decode_chunk(chunk_path: str, model=None, processor=None) -> str:
    """Decode one chunk path into placeholder text."""
    _ = model, processor
    chunk_name = os.path.basename(chunk_path)
    return f"transcript({chunk_name})"
