import json
import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import typer

from .extractor import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    create_client,
    extract_trivia,
    prepare_estimate,
)
from .transcriber import transcribe_audio, WhisperXEncoder

app = typer.Typer(help="ayqm-transcribe: A tool for transcribing audio and extracting trivia questions.")


def load_repo_env() -> None:
    load_dotenv(dotenv_path=Path(".env"))


@app.command()
def transcribe(
    audio_path: str = typer.Argument(..., help="Path to the audio file to transcribe."),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the output JSON file. Defaults to <audio_name>.json in the same folder."
    ),
    model_name: str = typer.Option(
        "base",
        "--model",
        "-m",
        help="Whisper model size/name to use (e.g., 'tiny', 'base', 'small', 'medium', 'large-v3')."
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Hardware device to use ('cpu', 'cuda', 'mps')."
    ),
    compute_type: str = typer.Option(
        "int8",
        "--compute-type",
        "-c",
        help="Precision type to use ('float16', 'int8', 'float32')."
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Batch size for transcription execution."
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Enable speaker diarization. Requires Hugging Face agreement and token."
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="Hugging Face User Access Token. Can also be set via HF_TOKEN environment variable."
    ),
    min_speakers: Optional[int] = typer.Option(
        None,
        "--min-speakers",
        help="Hint for minimum number of speakers (used if diarization is enabled)."
    ),
    max_speakers: Optional[int] = typer.Option(
        None,
        "--max-speakers",
        help="Hint for maximum number of speakers (used if diarization is enabled)."
    ),
):
    """
    Transcribes an audio file locally using WhisperX, performs word alignment,
    and optionally executes speaker diarization. Saves output as JSON.
    """
    load_repo_env()
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    resolved_token = hf_token or os.environ.get("HF_TOKEN")

    if not output:
        path_obj = Path(audio_path)
        output = str(path_obj.with_suffix(".json"))

    typer.echo(f"Transcribing {audio_path}...")
    try:
        result = transcribe_audio(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            diarize=diarize,
            hf_token=resolved_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        typer.echo(f"Writing transcript to {output}...")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, cls=WhisperXEncoder, indent=2, ensure_ascii=False)
        
        typer.echo("Transcription completed successfully!")
    except Exception as e:
        typer.echo(f"Error during transcription: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def extract(
    transcript_path: str = typer.Argument(..., help="Path to a WhisperX transcript JSON file."),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the output trivia JSON file. Defaults to <transcript_name>.trivia.json.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=f"Gemini model to use for trivia extraction. Defaults to AYQM_GEMINI_MODEL or {DEFAULT_MODEL}.",
    ),
    max_output_tokens: int = typer.Option(
        DEFAULT_MAX_OUTPUT_TOKENS,
        "--max-output-tokens",
        help="Maximum output tokens for the extraction response.",
    ),
    estimate_only: bool = typer.Option(
        False,
        "--estimate-only",
        help="Only estimate exact input tokens and input-token cost; do not run extraction.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive approval prompt after showing the input-token cost estimate.",
    ),
):
    """
    Extracts structured trivia from a transcript JSON file using Gemini.
    """
    load_repo_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    resolved_model = model or os.environ.get("AYQM_GEMINI_MODEL") or DEFAULT_MODEL

    if not output:
        output = str(Path(transcript_path).with_suffix(".trivia.json"))

    try:
        client = create_client()
        estimate, _prompt = prepare_estimate(transcript_path, client=client, model=resolved_model)
        typer.echo(f"Model: {resolved_model}")
        typer.echo(f"Exact input tokens: {estimate.input_tokens}")
        typer.echo(f"Estimated input-token cost: ${estimate.estimated_input_cost_usd:.6f} USD")
        typer.echo("Output-token cost is not included in this pre-approval estimate.")

        if estimate_only:
            raise typer.Exit(code=0)

        if not yes and not typer.confirm("Run trivia extraction with this input-token cost estimate?"):
            typer.echo("Extraction cancelled.")
            raise typer.Exit(code=1)

        result = extract_trivia(
            transcript_path,
            client=client,
            model=resolved_model,
            max_output_tokens=max_output_tokens,
            estimated_usage=estimate,
        )

        typer.echo(f"Writing trivia to {output}...")
        with open(output, "w", encoding="utf-8") as output_file:
            json.dump(result.model_dump(), output_file, indent=2, ensure_ascii=False)
        typer.echo(f"Extracted {len(result.trivia)} trivia items.")
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error during trivia extraction: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
