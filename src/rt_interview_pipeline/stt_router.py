from __future__ import annotations

import time
from dataclasses import dataclass
from string import printable
from typing import Protocol

from .errors import PipelineError
from .models import AudioChunk, AudioQualityProfile, PipelineMode, TranscriptCandidate
from .retry import with_backoff


class STTProvider(Protocol):
    name: str
    expected_latency_ms: int
    cost_per_minute_usd: float
    available: bool

    def transcribe(self, chunks: list[AudioChunk]) -> TranscriptCandidate:
        ...


@dataclass
class MockSTTProvider:
    name: str
    expected_latency_ms: int
    cost_per_minute_usd: float
    confidence_bias: float = 0.0
    available: bool = True

    def transcribe(self, chunks: list[AudioChunk]) -> TranscriptCandidate:
        if not self.available:
            raise PipelineError("stt_router", f"{self.name} is unavailable", "Speech service is unavailable.")

        started = time.perf_counter()
        raw = b" ".join(chunk.data for chunk in chunks)
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text or text.startswith("RIFF") or self._non_printable_ratio(text) > 0.15:
            text = "Explain how you would design a real-time interview assistant."

        confidence = max(0.0, min(0.99, 0.78 + self.confidence_bias))
        latency_ms = int((time.perf_counter() - started) * 1000) + self.expected_latency_ms
        return TranscriptCandidate(
            provider=self.name,
            text=" ".join(text.split()),
            confidence=confidence,
            latency_ms=latency_ms,
            cost_usd=self.cost_per_minute_usd,
            metadata={"mock": True},
        )

    def _non_printable_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        allowed = set(printable)
        return sum(1 for character in text if character not in allowed) / len(text)


class STTRouter:
    def __init__(self, providers: list[STTProvider] | None = None) -> None:
        self.providers = providers or [
            MockSTTProvider("openai_whisper_gpt_4o_transcribe", 700, 0.006, 0.07),
            MockSTTProvider("google_speech_to_text", 620, 0.008, 0.03),
            MockSTTProvider("aws_transcribe", 850, 0.007, 0.01),
            MockSTTProvider("deepgram_nova", 430, 0.005, 0.04),
            MockSTTProvider("local_faster_whisper", 300, 0.0, -0.03),
        ]

    def route(
        self,
        chunks: list[AudioChunk],
        profile: AudioQualityProfile,
        *,
        mode: PipelineMode = "smart",
    ) -> list[TranscriptCandidate]:
        available = [provider for provider in self.providers if provider.available]
        if not available:
            raise PipelineError("stt_router", "No STT providers available", "No speech transcription service is available.")

        providers = available if mode == "benchmark" else [self._select_best_provider(available, profile)]
        candidates: list[TranscriptCandidate] = []
        errors: list[str] = []

        for provider in providers:
            try:
                candidates.append(with_backoff(lambda provider=provider: provider.transcribe(chunks), attempts=2))
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                if mode == "smart":
                    fallback = self._fallback_provider(available, provider.name)
                    if fallback:
                        candidates.append(with_backoff(lambda fallback=fallback: fallback.transcribe(chunks), attempts=2))

        if not candidates:
            raise PipelineError(
                "stt_router",
                "All STT providers failed",
                "Could not transcribe the audio. Try again or use a different audio file.",
                details={"provider_errors": errors},
            )

        return candidates

    def _select_best_provider(self, providers: list[STTProvider], profile: AudioQualityProfile) -> STTProvider:
        def provider_score(provider: STTProvider) -> float:
            latency_score = 1.0 / max(provider.expected_latency_ms, 1)
            cost_score = 1.0 / (provider.cost_per_minute_usd + 0.001)
            quality_weight = 1.2 if profile.clarity_score < 0.55 and "openai" in provider.name else 1.0
            local_weight = 1.15 if profile.clarity_score > 0.8 and provider.cost_per_minute_usd == 0 else 1.0
            return quality_weight * local_weight * (latency_score * 450 + cost_score * 0.01)

        return max(providers, key=provider_score)

    def _fallback_provider(self, providers: list[STTProvider], failed_name: str) -> STTProvider | None:
        ordered = sorted(providers, key=lambda provider: (provider.cost_per_minute_usd, provider.expected_latency_ms))
        return next((provider for provider in ordered if provider.name != failed_name), None)
