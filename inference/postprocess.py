"""Post-processing helpers for chunk transcripts."""


def stitch_chunks(transcripts: list[str]) -> str:
    """Join chunk transcripts and normalize whitespace."""
    joined_text = " ".join(text.strip() for text in transcripts if text)
    return " ".join(joined_text.split())
