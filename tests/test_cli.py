import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from ayqm_transcribe import cli
from ayqm_transcribe.extractor import EstimatedUsage, ExtractionOutput, UsageReport


class ProcessCommandTests(unittest.TestCase):
    def test_process_estimate_only_writes_transcript_and_skips_extraction(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "episode.mp3"
            audio_path.write_bytes(b"audio")

            with (
                patch.object(cli, "transcribe_audio", return_value={"segments": []}) as transcribe_mock,
                patch.object(cli, "create_client", return_value=SimpleNamespace()),
                patch.object(
                    cli,
                    "prepare_estimate",
                    return_value=(EstimatedUsage(input_tokens=1000, estimated_input_cost_usd=0.00025), "prompt"),
                ),
                patch.object(cli, "extract_trivia") as extract_mock,
            ):
                result = runner.invoke(cli.app, ["process", str(audio_path), "--estimate-only"])

            transcript_path = Path(tmpdir) / "episode.json"
            trivia_path = Path(tmpdir) / "episode.trivia.json"

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(transcript_path.exists())
            self.assertFalse(trivia_path.exists())
            self.assertEqual(json.loads(transcript_path.read_text()), {"segments": []})
            self.assertIn("Exact input tokens: 1000", result.output)
            transcribe_mock.assert_called_once()
            extract_mock.assert_not_called()

    def test_process_yes_writes_transcript_and_trivia(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "episode.mp3"
            audio_path.write_bytes(b"audio")
            trivia_output = Path(tmpdir) / "out.trivia.json"
            extraction = ExtractionOutput(
                source_transcript=str(Path(tmpdir) / "episode.json"),
                model="gemini-3.1-flash-lite",
                usage=UsageReport(
                    estimated=EstimatedUsage(input_tokens=1000, estimated_input_cost_usd=0.00025)
                ),
                trivia=[],
            )

            with (
                patch.object(cli, "transcribe_audio", return_value={"segments": []}),
                patch.object(cli, "create_client", return_value=SimpleNamespace()),
                patch.object(
                    cli,
                    "prepare_estimate",
                    return_value=(EstimatedUsage(input_tokens=1000, estimated_input_cost_usd=0.00025), "prompt"),
                ),
                patch.object(cli, "extract_trivia", return_value=extraction) as extract_mock,
            ):
                result = runner.invoke(
                    cli.app,
                    ["process", str(audio_path), "--trivia-output", str(trivia_output), "--yes"],
                )

            transcript_path = Path(tmpdir) / "episode.json"

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(transcript_path.exists())
            self.assertTrue(trivia_output.exists())
            self.assertEqual(json.loads(trivia_output.read_text())["trivia"], [])
            self.assertIn("Extracted 0 trivia items.", result.output)
            extract_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
