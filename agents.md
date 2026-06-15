# Project: ayqm-transcribe

## Overview
`ayqm-transcribe` is a tool for transcribing podcast audio (specifically trivia shows) and extracting trivia questions. It is designed to be used as both a Python library and a command-line interface (CLI).

## Goal
- Transcribe audio with 2-3 speakers.
- Handle conversational banter and tangential trivia.
- Extract structured trivia from transcripts, including asked trivia questions and trivia facts mentioned in conversation.
- For each trivia item, include question, answer, keywords, timestamps, speaker diarization, and whether it was asked or only mentioned.
- Do not extract hints, transcript excerpts, or segment ids in trivia output.

## Tech Stack
- **Language**: Python
- **Package Manager**: `uv`
- **Architecture**: CLI & Library

## Operational Rules
- **Dependencies**: Never run `uv add` commands directly. Always propose the dependencies first and wait for the user to execute or approve.
- **Shell Commands**: Always explain the purpose and impact of any shell command before proposing to run it.
- **UV Path**: Use `/Users/adityasrivastava/.local/bin/uv` for all `uv` related tasks.

## Local Development & Transcription Notes
- **WhisperX on macOS**: WhisperX operates most stably on macOS/CPU. Default configuration should use `--device cpu` and `--compute-type int8` to avoid PyTorch MPS compatibility/segment alignment crashes.
- **UV Dependency Locking**: To prevent multi-platform dependency conflicts with PyTorch CUDA indices, `pyproject.toml` restricts universal resolution specifically to macOS via `tool.uv.environments = ["sys_platform == 'darwin'"]`.
- **Speaker Diarization**: Enabling speaker diarization (the `--diarize` flag) requires:
  1. A Hugging Face account and user agreement approvals on HF for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
  2. Setting the `HF_TOKEN` environment variable or passing `--hf-token`.
- **Trivia Extraction**: Extraction uses Gemini via `google-genai`. Load credentials from `.env`; prefer `GEMINI_API_KEY`, with `GOOGLE_API_KEY` supported as a fallback.
- **Cost Approval**: The `extract` command always counts exact input tokens first and shows an input-token-only cost estimate before generation. Output-token cost is intentionally excluded from the pre-approval estimate, but actual post-run usage includes output tokens when Gemini returns usage metadata.
- **Default Gemini Model**: Use `gemini-3.1-flash-lite` by default, or override with `AYQM_GEMINI_MODEL` or `--model`.

## Common Commands
- Transcribe an audio file:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py transcribe test_audio/130.mp3
  ```
- Transcribe with diarization:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py transcribe test_audio/130.mp3 --diarize --min-speakers 2 --max-speakers 3
  ```
- Estimate trivia extraction input tokens and input-token cost without generation:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py extract test_audio/SonsOfGnadhiLegionnairesDisease.json --estimate-only
  ```
- Extract trivia with interactive cost approval:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py extract test_audio/SonsOfGnadhiLegionnairesDisease.json
  ```
- Extract trivia without the approval prompt after showing the estimate:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py extract test_audio/SonsOfGnadhiLegionnairesDisease.json --yes
  ```
- Transcribe audio and extract trivia in one command:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py process test_audio/130.mp3
  ```
- Transcribe audio, save the transcript, estimate extraction cost, and stop before Gemini generation:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py process test_audio/130.mp3 --estimate-only
  ```
- Transcribe audio and extract trivia without the approval prompt after showing the estimate:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python main.py process test_audio/130.mp3 --yes
  ```
- Run tests:
  ```sh
  /Users/adityasrivastava/.local/bin/uv run python -m unittest discover -s tests
  ```

## Progress & Context
- [x] Project initialized with `uv`.
- [x] Initial `agents.md` created with operational rules.
- [x] Installed basic dependencies: `typer`, `pydantic`.
- [x] Implement audio transcription.
- [x] Implement trivia extraction.
- [x] Implement CLI interface.
