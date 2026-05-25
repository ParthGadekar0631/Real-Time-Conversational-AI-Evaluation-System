from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from .answer_generation import AnswerGenerationModule
from .audio_analysis import AudioAnalysisModule
from .audio_input import AudioInputLayer
from .errors import ErrorRecord, PipelineError
from .llm_router import LLMRouter
from .models import AnswerMode, PipelineMode, PipelineResult
from .microphone_input import MicrophoneInputLayer
from .storage import JsonSessionStorage
from .stt_router import STTRouter
from .transcript_evaluation import TranscriptEvaluationEngine


class InterviewPipeline:
    def __init__(
        self,
        *,
        audio_input: AudioInputLayer | None = None,
        audio_analysis: AudioAnalysisModule | None = None,
        stt_router: STTRouter | None = None,
        transcript_evaluator: TranscriptEvaluationEngine | None = None,
        llm_router: LLMRouter | None = None,
        answer_generator: AnswerGenerationModule | None = None,
        storage: JsonSessionStorage | None = None,
        microphone_input: MicrophoneInputLayer | None = None,
    ) -> None:
        self.audio_input = audio_input or AudioInputLayer()
        self.audio_analysis = audio_analysis or AudioAnalysisModule()
        self.stt_router = stt_router or STTRouter()
        self.transcript_evaluator = transcript_evaluator or TranscriptEvaluationEngine()
        self.llm_router = llm_router or LLMRouter()
        self.answer_generator = answer_generator or AnswerGenerationModule()
        self.storage = storage or JsonSessionStorage()
        self.microphone_input = microphone_input or MicrophoneInputLayer()

    def run_file(
        self,
        audio_file: str | Path,
        *,
        mode: PipelineMode = "smart",
        answer_mode: AnswerMode = "short",
    ) -> PipelineResult:
        return self._run_from_source(lambda: self.audio_input.load_file(audio_file), mode=mode, answer_mode=answer_mode)

    def run_microphone(
        self,
        *,
        seconds: int = 8,
        mode: PipelineMode = "smart",
        answer_mode: AnswerMode = "short",
    ) -> PipelineResult:
        return self._run_from_source(lambda: self.microphone_input.record(seconds=seconds), mode=mode, answer_mode=answer_mode)

    def _run_from_source(
        self,
        load_chunks,
        *,
        mode: PipelineMode = "smart",
        answer_mode: AnswerMode = "short",
    ) -> PipelineResult:
        result = PipelineResult.empty(session_id=uuid4().hex)
        started_total = time.perf_counter()

        try:
            chunks, result.latency_breakdown_ms["audio_input"] = self._timed(load_chunks)
            profile, result.latency_breakdown_ms["audio_analysis"] = self._timed(lambda: self.audio_analysis.analyze(chunks))
            result.audio_quality = profile

            candidates, result.latency_breakdown_ms["stt"] = self._timed(
                lambda: self.stt_router.route(chunks, profile, mode=mode)
            )
            transcript, result.latency_breakdown_ms["transcript_evaluation"] = self._timed(
                lambda: self.transcript_evaluator.select_best(candidates)
            )
            result.transcript = transcript
            result.selected_stt_model = transcript.provider

            question, result.latency_breakdown_ms["question_analysis"] = self._timed(
                lambda: self.llm_router.classify(transcript.text)
            )
            result.question = question

            llm_response, result.latency_breakdown_ms["llm"] = self._timed(
                lambda: self.llm_router.route(question, detailed=answer_mode == "detailed")
            )
            answer, result.latency_breakdown_ms["answer_generation"] = self._timed(
                lambda: self.answer_generator.post_process(llm_response)
            )
            result.answer = answer
            result.selected_llm_model = answer.provider
        except PipelineError as exc:
            result.errors.append(exc.to_record())
        except Exception as exc:
            result.errors.append(
                ErrorRecord(
                    stage="global",
                    message=str(exc),
                    user_message="An unexpected error occurred. Partial results were saved when possible.",
                    recoverable=False,
                )
            )
        finally:
            result.latency_breakdown_ms["total"] = int((time.perf_counter() - started_total) * 1000)
            try:
                result.log_path = self.storage.save(result)
            except PipelineError as exc:
                result.errors.append(exc.to_record())

        return result

    def _timed(self, operation):
        started = time.perf_counter()
        output = operation()
        return output, int((time.perf_counter() - started) * 1000)
