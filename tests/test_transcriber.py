import sys
import types
import unittest
from unittest.mock import patch

from ayqm_transcribe import transcriber


class DiarizationPipelineCompatibilityTests(unittest.TestCase):
    def test_create_diarization_pipeline_uses_token_argument(self):
        calls = []

        class FakePipeline:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        with patch.object(transcriber.whisperx, "DiarizationPipeline", FakePipeline, create=True):
            pipeline = transcriber.create_diarization_pipeline("hf-token", "cpu")

        self.assertIsInstance(pipeline, FakePipeline)
        self.assertEqual(calls, [{"token": "hf-token", "device": "cpu"}])

    def test_create_diarization_pipeline_falls_back_to_use_auth_token(self):
        calls = []

        class FakePipeline:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "token" in kwargs:
                    raise TypeError("__init__() got an unexpected keyword argument 'token'")

        with patch.object(transcriber.whisperx, "DiarizationPipeline", FakePipeline, create=True):
            pipeline = transcriber.create_diarization_pipeline("hf-token", "cpu")

        self.assertIsInstance(pipeline, FakePipeline)
        self.assertEqual(
            calls,
            [
                {"token": "hf-token", "device": "cpu"},
                {"use_auth_token": "hf-token", "device": "cpu"},
            ],
        )

    def test_get_diarization_pipeline_class_uses_diarize_module_when_not_exported(self):
        class FakePipeline:
            pass

        fake_diarize = types.ModuleType("whisperx.diarize")
        fake_diarize.DiarizationPipeline = FakePipeline

        with (
            patch.object(transcriber.whisperx, "DiarizationPipeline", None, create=True),
            patch.dict(sys.modules, {"whisperx.diarize": fake_diarize}),
        ):
            pipeline_class = transcriber.get_diarization_pipeline_class()

        self.assertIs(pipeline_class, FakePipeline)


if __name__ == "__main__":
    unittest.main()
