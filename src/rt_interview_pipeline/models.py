from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

QuestionType = Literal["coding_sql", "conceptual", "factual", "general"]
PipelineMode = Literal["smart", "benchmark"]
AnswerMode = Literal["short", "detailed"]


@dataclass
class AudioChunk:
    index: int
    data: bytes
    duration_ms: int


@dataclass
class AudioQualityProfile:
    volume_db: float
    noise_snr: float
    silence_ratio: float
    speech_duration_ms: int
    clarity_score: float


@dataclass
class TranscriptCandidate:
    provider: str
    text: str
    confidence: float
    latency_ms: int
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class FinalTranscript:
    text: str
    provider: str
    confidence: float
    score: float
    candidates: list[TranscriptCandidate]


@dataclass
class InterviewQuestion:
    text: str
    question_type: QuestionType


@dataclass
class LLMResponse:
    provider: str
    text: str
    latency_ms: int
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    session_id: str
    created_at: str
    audio_quality: AudioQualityProfile | None
    transcript: FinalTranscript | None
    question: InterviewQuestion | None
    answer: LLMResponse | None
    selected_stt_model: str | None
    selected_llm_model: str | None
    latency_breakdown_ms: dict[str, int]
    errors: list[Any] = field(default_factory=list)
    log_path: str | None = None

    @classmethod
    def empty(cls, session_id: str) -> "PipelineResult":
        return cls(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            audio_quality=None,
            transcript=None,
            question=None,
            answer=None,
            selected_stt_model=None,
            selected_llm_model=None,
            latency_breakdown_ms={},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
