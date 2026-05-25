from __future__ import annotations

import re

from .errors import PipelineError
from .models import FinalTranscript, TranscriptCandidate


class TranscriptEvaluationEngine:
    def select_best(self, candidates: list[TranscriptCandidate]) -> FinalTranscript:
        valid = [candidate for candidate in candidates if candidate.text.strip()]
        if not valid:
            raise PipelineError("transcript_evaluation", "No valid transcript", "No valid transcript was produced.")

        for candidate in valid:
            candidate.score = self._score(candidate, valid)

        best = max(valid, key=lambda candidate: candidate.score)
        return FinalTranscript(
            text=best.text,
            provider=best.provider,
            confidence=best.confidence,
            score=best.score,
            candidates=valid,
        )

    def _score(self, candidate: TranscriptCandidate, all_candidates: list[TranscriptCandidate]) -> float:
        words = self._words(candidate.text)
        overlap = self._average_overlap(words, all_candidates)
        completeness = min(1.0, len(words) / 12.0)
        grammar = 0.7 if candidate.text and candidate.text[0].isupper() else 0.55
        latency = max(0.0, 1.0 - candidate.latency_ms / 5_000)
        cost = max(0.0, 1.0 - candidate.cost_usd / 0.05)
        return round(
            candidate.confidence * 0.34
            + overlap * 0.2
            + completeness * 0.18
            + grammar * 0.1
            + latency * 0.1
            + cost * 0.08,
            3,
        )

    def _average_overlap(self, words: set[str], candidates: list[TranscriptCandidate]) -> float:
        if len(candidates) <= 1:
            return 1.0
        scores = []
        for candidate in candidates:
            other_words = self._words(candidate.text)
            if other_words == words:
                continue
            union = words | other_words
            scores.append(len(words & other_words) / len(union) if union else 0.0)
        return sum(scores) / len(scores) if scores else 1.0

    def _words(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
