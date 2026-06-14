import json
import unittest
from types import SimpleNamespace

from ayqm_transcribe.extractor import (
    actual_usage_from_response,
    estimate_usage,
    format_range,
    normalize_segments,
    parse_extraction_response,
)


class ExtractorTests(unittest.TestCase):
    def test_normalize_segments_keeps_timestamps_and_speaker(self):
        transcript = {
            "segments": [
                {"start": 1.2, "end": 4.7, "speaker": "SPEAKER_00", "text": " A question? "},
                {"start": 5.0, "end": 6.0, "text": ""},
            ]
        }

        segments = normalize_segments(transcript)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["display"], "00:00:01-00:00:05")
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")
        self.assertEqual(segments[0]["text"], "A question?")

    def test_format_range(self):
        self.assertEqual(format_range(123.4, 145.6), "00:02:03-00:02:26")

    def test_parse_response_has_no_hints_or_excerpts(self):
        payload = {
            "trivia": [
                {
                    "id": "trivia_001",
                    "type": "mentioned_trivia",
                    "question": "What happened?",
                    "answer": "A fact was mentioned.",
                    "keywords": ["fact"],
                    "timestamps": {
                        "start": 1.0,
                        "end": 2.0,
                        "display": "00:00:01-00:00:02",
                    },
                    "speaker_diarization": {
                        "asker_speaker": None,
                        "answerer_speakers": [],
                        "mentioned_by_speakers": ["SPEAKER_00"],
                    },
                    "confidence": "high",
                }
            ]
        }

        extraction = parse_extraction_response(SimpleNamespace(text=json.dumps(payload)))
        dumped = extraction.model_dump()

        self.assertEqual(len(extraction.trivia), 1)
        self.assertNotIn("hints", dumped["trivia"][0])
        self.assertNotIn("source", dumped["trivia"][0])
        self.assertNotIn("segment_ids", json.dumps(dumped))
        self.assertNotIn("excerpt", json.dumps(dumped))

    def test_actual_usage_cost_includes_output_after_run(self):
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1_000_000,
                candidates_token_count=1_000_000,
                thoughts_token_count=0,
                total_token_count=2_000_000,
            )
        )

        usage = actual_usage_from_response(response)

        self.assertEqual(usage.prompt_token_count, 1_000_000)
        self.assertEqual(usage.candidates_token_count, 1_000_000)
        self.assertEqual(usage.actual_cost_usd, 1.75)

    def test_estimate_usage_uses_input_tokens_only(self):
        class FakeModels:
            def count_tokens(self, **kwargs):
                return SimpleNamespace(total_tokens=1_000_000)

        client = SimpleNamespace(models=FakeModels())

        estimate = estimate_usage(client, "gemini-3.1-flash-lite", "prompt")

        self.assertEqual(estimate.input_tokens, 1_000_000)
        self.assertEqual(estimate.estimated_input_cost_usd, 0.25)


if __name__ == "__main__":
    unittest.main()
