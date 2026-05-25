from __future__ import annotations

import io
import time
from dataclasses import dataclass

from .errors import PipelineError
from .models import AudioChunk, InterviewQuestion, LLMResponse, QuestionType, TranscriptCandidate


def _require_openai_client(api_key: str | None):
    if not api_key:
        raise PipelineError(
            "config",
            "OPENAI_API_KEY is not configured",
            "Set OPENAI_API_KEY in your .env file.",
            recoverable=False,
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PipelineError(
            "config",
            "OpenAI SDK is not installed",
            "Install project dependencies before using OpenAI providers.",
            recoverable=False,
        ) from exc
    return OpenAI(api_key=api_key)


@dataclass
class OpenAISTTProvider:
    api_key: str | None
    model: str = "gpt-4o-transcribe"
    name: str = "openai_gpt_4o_transcribe"
    expected_latency_ms: int = 900
    cost_per_minute_usd: float = 0.006
    available: bool = True

    def transcribe(self, chunks: list[AudioChunk]) -> TranscriptCandidate:
        client = _require_openai_client(self.api_key)
        started = time.perf_counter()
        audio_bytes = b"".join(chunk.data for chunk in chunks)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"

        try:
            transcription = client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="json",
            )
        except Exception as exc:
            raise PipelineError(
                "stt_router",
                f"OpenAI transcription failed: {exc}",
                "OpenAI transcription failed. Check your API key, quota, model name, and audio input.",
            ) from exc

        text = getattr(transcription, "text", "") or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not text.strip():
            raise PipelineError("stt_router", "OpenAI returned an empty transcript", "No speech was detected.")

        return TranscriptCandidate(
            provider=self.name,
            text=" ".join(text.split()),
            confidence=0.9,
            latency_ms=latency_ms,
            cost_usd=self.cost_per_minute_usd,
            metadata={"model": self.model},
        )


@dataclass
class OpenAILLMProvider:
    api_key: str | None
    model: str = "gpt-4.1-mini"
    name: str = "openai_gpt"
    strengths: set[QuestionType] | None = None
    expected_latency_ms: int = 900
    cost_rank: int = 2
    available: bool = True

    def __post_init__(self) -> None:
        if self.strengths is None:
            self.strengths = {"coding_sql", "conceptual", "factual", "general"}

    def generate(self, question: InterviewQuestion, *, detailed: bool) -> LLMResponse:
        client = _require_openai_client(self.api_key)
        started = time.perf_counter()
        style = "Give a concise interview-ready answer." if not detailed else "Give a detailed interview-ready explanation."
        prompt = (
            f"{style}\n\n"
            f"Question type: {question.question_type}\n"
            f"Question/transcript:\n{question.text}\n\n"
            "Answer clearly, avoid filler, and include practical tradeoffs when useful."
        )

        try:
            response = client.responses.create(model=self.model, input=prompt)
        except Exception as exc:
            raise PipelineError(
                "llm_router",
                f"OpenAI response failed: {exc}",
                "OpenAI answer generation failed. Check your API key, quota, and model name.",
            ) from exc

        text = getattr(response, "output_text", "") or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not text.strip():
            raise PipelineError("llm_router", "OpenAI returned an empty response", "The AI model returned an empty answer.")

        return LLMResponse(
            provider=self.name,
            text=text,
            latency_ms=latency_ms,
            metadata={"model": self.model},
        )
