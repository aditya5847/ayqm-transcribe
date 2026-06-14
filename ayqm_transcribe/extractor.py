import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_INPUT_PRICE_PER_1M = 0.25
DEFAULT_OUTPUT_PRICE_PER_1M = 1.50


class TimestampRange(BaseModel):
    start: float
    end: float
    display: str


class SpeakerDiarization(BaseModel):
    asker_speaker: Optional[str] = None
    answerer_speakers: list[str] = Field(default_factory=list)
    mentioned_by_speakers: list[str] = Field(default_factory=list)


class TriviaItem(BaseModel):
    id: str
    type: Literal["asked_question", "mentioned_trivia"]
    question: Optional[str] = None
    answer: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    timestamps: TimestampRange
    speaker_diarization: SpeakerDiarization = Field(default_factory=SpeakerDiarization)
    confidence: Literal["low", "medium", "high"] = "medium"

    @field_validator("keywords")
    @classmethod
    def trim_keywords(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class TriviaExtraction(BaseModel):
    trivia: list[TriviaItem] = Field(default_factory=list)


class EstimatedUsage(BaseModel):
    input_tokens: int
    estimated_input_cost_usd: float


class ActualUsage(BaseModel):
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0
    actual_cost_usd: float = 0.0


class UsageReport(BaseModel):
    estimated: EstimatedUsage
    actual: Optional[ActualUsage] = None


class ExtractionOutput(BaseModel):
    source_transcript: str
    model: str
    usage: UsageReport
    trivia: list[TriviaItem]


def resolve_api_key(api_key: Optional[str] = None) -> str:
    resolved = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file.")
    return resolved


def get_input_price_per_1m() -> float:
    return float(os.environ.get("AYQM_GEMINI_INPUT_PRICE_PER_1M", DEFAULT_INPUT_PRICE_PER_1M))


def get_output_price_per_1m() -> float:
    return float(os.environ.get("AYQM_GEMINI_OUTPUT_PRICE_PER_1M", DEFAULT_OUTPUT_PRICE_PER_1M))


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_range(start: float, end: float) -> str:
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


def load_transcript(transcript_path: str | Path) -> dict[str, Any]:
    path = Path(transcript_path)
    with path.open("r", encoding="utf-8") as transcript_file:
        data = json.load(transcript_file)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Transcript JSON must be an object with a 'segments' list.")
    return data


def normalize_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(transcript.get("segments", [])):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        normalized.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "display": format_range(start, end),
                "speaker": segment.get("speaker"),
                "text": text,
            }
        )
    return normalized


def build_prompt(normalized_segments: list[dict[str, Any]]) -> str:
    transcript_lines = []
    for segment in normalized_segments:
        speaker = segment["speaker"] or "UNKNOWN"
        transcript_lines.append(
            f'[{segment["display"]}] speaker={speaker}: {segment["text"]}'
        )

    transcript_text = "\n".join(transcript_lines)
    return f"""Extract trivia from this podcast transcript.

Return high-recall trivia items of two types:
- asked_question: a quiz/trivia question actually asked in the episode.
- mentioned_trivia: a self-contained factual claim mentioned in conversation that could plausibly become a quiz question.

Do not extract hints. Do not include transcript excerpts. Do not include segment ids.
For asked questions, preserve the question as asked when possible and include the answer if it appears in the transcript.
For mentioned trivia, write a concise implied quiz-style question when useful, and put the mentioned fact in answer.
Use transcript-relative timestamps from the bracketed ranges. If an item spans multiple nearby lines, use the earliest start and latest end.
Speaker fields must use only explicit speaker labels from the transcript. If speakers are UNKNOWN, use null or empty lists.
Keep keywords short and useful for search.

Transcript:
{transcript_text}
"""


def build_generation_config(max_output_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=TriviaExtraction,
    )


def build_count_config() -> types.CountTokensConfig:
    return types.CountTokensConfig()


def create_client(api_key: Optional[str] = None) -> genai.Client:
    return genai.Client(api_key=resolve_api_key(api_key))


def estimate_usage(client: genai.Client, model: str, prompt: str) -> EstimatedUsage:
    response = client.models.count_tokens(
        model=model,
        contents=prompt,
        config=build_count_config(),
    )
    input_tokens = int(getattr(response, "total_tokens", 0) or 0)
    cost = input_tokens / 1_000_000 * get_input_price_per_1m()
    return EstimatedUsage(
        input_tokens=input_tokens,
        estimated_input_cost_usd=round(cost, 6),
    )


def actual_usage_from_response(response: Any) -> ActualUsage:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return ActualUsage()

    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    cost = (
        prompt_tokens / 1_000_000 * get_input_price_per_1m()
        + (candidate_tokens + thoughts_tokens) / 1_000_000 * get_output_price_per_1m()
    )
    return ActualUsage(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
        thoughts_token_count=thoughts_tokens,
        total_token_count=total_tokens,
        actual_cost_usd=round(cost, 6),
    )


def parse_extraction_response(response: Any) -> TriviaExtraction:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, TriviaExtraction):
        return parsed
    if isinstance(parsed, dict):
        return TriviaExtraction.model_validate(parsed)

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini response did not include structured JSON text.")
    return TriviaExtraction.model_validate_json(text)


def extract_trivia(
    transcript_path: str | Path,
    *,
    client: genai.Client,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    estimated_usage: Optional[EstimatedUsage] = None,
) -> ExtractionOutput:
    transcript = load_transcript(transcript_path)
    normalized_segments = normalize_segments(transcript)
    prompt = build_prompt(normalized_segments)
    estimated = estimated_usage or estimate_usage(client, model, prompt)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=build_generation_config(max_output_tokens),
    )
    extraction = parse_extraction_response(response)
    actual_usage = actual_usage_from_response(response)

    return ExtractionOutput(
        source_transcript=str(transcript_path),
        model=model,
        usage=UsageReport(estimated=estimated, actual=actual_usage),
        trivia=extraction.trivia,
    )


def prepare_estimate(
    transcript_path: str | Path,
    *,
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> tuple[EstimatedUsage, str]:
    transcript = load_transcript(transcript_path)
    prompt = build_prompt(normalize_segments(transcript))
    return estimate_usage(client, model, prompt), prompt
