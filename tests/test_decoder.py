from unittest.mock import Mock

import torch

from inference.decoder import decode_logits


def test_decode_logits_uses_processor_batch_decode_and_returns_text():
    processor = Mock()
    processor.batch_decode.return_value = ["Muraho"]
    logits = torch.randn(1, 20, 5)

    transcription = decode_logits(logits, processor)

    assert transcription == "Muraho"
    processor.batch_decode.assert_called_once()


def test_decode_logits_blank_output_returns_empty_string():
    processor = Mock()
    processor.batch_decode.return_value = [""]
    logits = torch.zeros(1, 4, 3)

    transcription = decode_logits(logits, processor)

    assert transcription == ""
