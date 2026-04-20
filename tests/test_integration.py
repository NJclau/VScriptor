"""Integration-style tests for chunk + stitch transcription flow."""

import numpy as np

import stt


class _FakeProcessor:
    def __call__(self, audio_array, **_kwargs):
        class _Inputs:
            input_values = np.zeros((1, len(audio_array)))

        return _Inputs()

    def batch_decode(self, predicted_ids):
        token_id = int(predicted_ids[0][0].item())
        mapping = {0: "muraho", 1: "amakuru", 2: "neza"}
        return [mapping.get(token_id, "")]


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, input_values):
        base = int(input_values.shape[-1] % 3)
        logits = np.zeros((1, 1, 3))
        logits[0, 0, base] = 1.0

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


class _FakeChunk:
    def __init__(self, width):
        self.width = width

    def squeeze(self, *_args, **_kwargs):
        return self

    def numpy(self):
        return np.zeros((self.width,))


def test_chunk_and_stitch_flow_matches_expected_join(monkeypatch):
    """Transcription should decode each chunk and stitch text with single spaces."""
    fake_chunks = [_FakeChunk(20), _FakeChunk(21), _FakeChunk(22)]

    monkeypatch.setattr(stt, "chunk_audio", lambda _path: fake_chunks)
    monkeypatch.setattr(stt, "get_model_and_processor", lambda: (_FakeModel(), _FakeProcessor()))

    transcription = stt.transcribe_long_audios("dummy.wav")
    assert transcription == "neza muraho amakuru"
