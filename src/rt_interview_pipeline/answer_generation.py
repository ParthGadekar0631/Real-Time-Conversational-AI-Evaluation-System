from __future__ import annotations

from .errors import PipelineError
from .models import LLMResponse


class AnswerGenerationModule:
    def post_process(self, response: LLMResponse, *, max_chars: int = 1_200) -> LLMResponse:
        text = " ".join(response.text.split())
        if not text:
            raise PipelineError("answer_generation", "Empty LLM response", "The AI response was empty.")

        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        response.text = text
        return response
