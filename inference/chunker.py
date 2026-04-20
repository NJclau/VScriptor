"""Chunk and streaming configuration for long audio decoding."""

import copy
import math

from omegaconf import OmegaConf

from config import CHUNK_DURATION_MS, OVERLAP_DURATION_MS

MODEL_STRIDE = 4
TOTAL_BUFFER_SECS = 30.0
BATCH_SIZE = 1
MILLIS_PER_SECOND = 1000


def build_streaming_chunk_config(asr_model) -> dict:
    """Build chunking and delay configuration derived from model preprocessor."""
    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    feature_stride = model_cfg.preprocessor["window_stride"]
    model_stride_secs = feature_stride * MODEL_STRIDE
    chunk_len_secs = CHUNK_DURATION_MS / MILLIS_PER_SECOND
    overlap_secs = OVERLAP_DURATION_MS / MILLIS_PER_SECOND
    left_right_context_secs = overlap_secs * 2
    total_buffer_secs = max(TOTAL_BUFFER_SECS, chunk_len_secs + left_right_context_secs)

    tokens_per_chunk = math.ceil(chunk_len_secs / model_stride_secs)
    mid_delay = math.ceil(
        (chunk_len_secs + (total_buffer_secs - chunk_len_secs) / 2) / model_stride_secs,
    )

    return {
        "batch_size": BATCH_SIZE,
        "chunk_len_secs": chunk_len_secs,
        "mid_delay": mid_delay,
        "model_stride_secs": model_stride_secs,
        "tokens_per_chunk": tokens_per_chunk,
        "total_buffer_secs": total_buffer_secs,
    }
