import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import whisperx

logger = logging.getLogger("ayqm_transcribe.transcriber")

class WhisperXEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely handle numpy scalar types (like np.float32 or np.int64)
    that WhisperX or faster-whisper might output in segments/words dictionaries.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 16,
    diarize: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Transcribes the given audio file using WhisperX.
    Aligns the transcript with word-level timestamps.
    Optionally performs speaker diarization if `diarize` is True and `hf_token` is provided.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_name: Name of the Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large-v3').
        device: Device to run inference on ('cpu', 'cuda', 'mps'). Default is 'cpu'.
        compute_type: Computation precision ('float16', 'int8', 'float32'). Default is 'int8' for CPU.
        batch_size: Batch size for WhisperX transcription.
        diarize: Whether to run speaker diarization.
        hf_token: Hugging Face User Access Token (required for diarization models).
        min_speakers: Optional minimum number of speakers.
        max_speakers: Optional maximum number of speakers.

    Returns:
        A dictionary containing the full transcription result, with segments and aligned words.
    """
    # 1. Load Audio
    logger.info(f"Loading audio file: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # 2. Transcription
    logger.info(f"Loading WhisperX model '{model_name}' on '{device}' with '{compute_type}'...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    logger.info(f"Transcribing audio (batch size: {batch_size})...")
    raw_result = model.transcribe(audio, batch_size=batch_size)

    # Free transcription model memory if we need to align on GPU
    # (Not strictly necessary on CPU, but good practice)
    del model

    # 3. Alignment
    detected_language = raw_result.get("language")
    logger.info(f"Detected language: {detected_language}. Loading alignment model...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        logger.info("Aligning transcription segments...")
        result = whisperx.align(
            raw_result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
    except Exception as e:
        logger.warning(f"Alignment failed: {e}. Falling back to unaligned segments.")
        result = raw_result

    # 4. Optional Diarization
    if diarize:
        if not hf_token:
            logger.error("Diarization requested but no Hugging Face token (hf_token) provided.")
            raise ValueError(
                "A Hugging Face access token (hf_token) is required for pyannote/speaker-diarization."
            )

        logger.info("Loading Diarization model...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        
        logger.info("Running speaker diarization...")
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        logger.info("Assigning speakers to transcription words...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result
