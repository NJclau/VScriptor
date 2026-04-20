"""Centralized configuration constants for transcription service."""

# Inference
MODEL_NAME = "badrex/w2v-bert-2.0-kinyarwanda-asr-1000h"
FALLBACK_MODEL_NAME = "stt_rw_conformer_ctc_large"
TARGET_SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 22_000
OVERLAP_DURATION_MS = 3_000
SILENCE_THRESH_DB = -40.0

# Scheduler
MAX_PRIMARY_SLOTS = 2
SHORT_JOB_THRESHOLD_SECS = 30.0
MAX_AUDIO_DURATION_SECS = 180.0

# State / Hub
HF_DATASET_REPO = "mbaza-nlp/stt-job-state"
JOB_STATE_FILE = "jobs.json"
CACHE_POLL_INTERVAL_SECS = 5

# Backoff
BACKOFF_BASE_SECS = 10
BACKOFF_MAX_SECS = 120
BACKOFF_MULTIPLIER = 2
