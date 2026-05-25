from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .errors import PipelineError
from .models import InterviewQuestion, LLMResponse, QuestionType
from .retry import with_backoff


class LLMProvider(Protocol):
    name: str
    strengths: set[QuestionType]
    expected_latency_ms: int
    cost_rank: int
    available: bool

    def generate(self, question: InterviewQuestion, *, detailed: bool) -> LLMResponse:
        ...


@dataclass
class MockLLMProvider:
    name: str
    strengths: set[QuestionType]
    expected_latency_ms: int
    cost_rank: int
    available: bool = True

    def generate(self, question: InterviewQuestion, *, detailed: bool) -> LLMResponse:
        if not self.available:
            raise PipelineError("llm_router", f"{self.name} is unavailable", "The selected AI model is unavailable.")

        started = time.perf_counter()
        prefix = "Detailed answer" if detailed else "Short answer"
        body = (
            f"{prefix}: {question.text} The best interview response should define the core idea, "
            "explain the tradeoffs, and close with a practical example."
        )
        latency_ms = int((time.perf_counter() - started) * 1000) + self.expected_latency_ms
        return LLMResponse(provider=self.name, text=body, latency_ms=latency_ms, metadata={"mock": True})


class LLMRouter:
    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        self.providers = providers or [
            MockLLMProvider("openai_gpt_series", {"coding_sql", "conceptual", "general"}, 900, 2),
            MockLLMProvider("claude_reasoning", {"coding_sql", "conceptual"}, 1_200, 3),
            MockLLMProvider("perplexity_search_backed", {"factual"}, 1_000, 2),
            MockLLMProvider("local_model", {"general"}, 450, 1),
        ]

    def classify(self, text: str) -> InterviewQuestion:
        lowered = text.lower()
        if any(token in lowered for token in ("sql", "code", "algorithm", "function", "leetcode")):
            question_type: QuestionType = "coding_sql"
        elif any(token in lowered for token in ("latest", "current", "today", "who is", "what is the date")):
            question_type = "factual"
        elif any(token in lowered for token in ("explain", "design", "why", "tradeoff", "concept")):
            question_type = "conceptual"
        else:
            question_type = "general"
        return InterviewQuestion(text=text, question_type=question_type)

    def route(self, question: InterviewQuestion, *, detailed: bool = False) -> LLMResponse:
        available = [provider for provider in self.providers if provider.available]
        if not available:
            raise PipelineError("llm_router", "No LLM providers available", "No AI response model is available.")

        provider = self._select_best_provider(available, question.question_type)
        try:
            return with_backoff(lambda: provider.generate(question, detailed=detailed), attempts=2)
        except Exception:
            fallback = self._fallback_provider(available, provider.name)
            if fallback is None:
                raise
            return with_backoff(lambda: fallback.generate(question, detailed=False), attempts=2)

    def _select_best_provider(self, providers: list[LLMProvider], question_type: QuestionType) -> LLMProvider:
        def provider_score(provider: LLMProvider) -> float:
            strength = 2.0 if question_type in provider.strengths else 0.7
            latency = 1.0 / max(provider.expected_latency_ms, 1)
            cost = 1.0 / provider.cost_rank
            return strength + latency * 300 + cost * 0.2

        return max(providers, key=provider_score)

    def _fallback_provider(self, providers: list[LLMProvider], failed_name: str) -> LLMProvider | None:
        ordered = sorted(providers, key=lambda provider: (provider.cost_rank, provider.expected_latency_ms))
        return next((provider for provider in ordered if provider.name != failed_name), None)
