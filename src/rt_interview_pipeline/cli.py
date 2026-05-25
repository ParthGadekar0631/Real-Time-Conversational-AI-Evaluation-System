from __future__ import annotations

import argparse

from .config import AppConfig
from .errors import PipelineError
from .llm_router import LLMRouter
from .microphone_input import MicrophoneInputLayer
from .openai_providers import OpenAILLMProvider, OpenAISTTProvider
from .pipeline import InterviewPipeline
from .postgres_storage import PostgresSessionStorage
from .storage import JsonSessionStorage
from .stt_router import STTRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adaptive real-time AI interview response pipeline.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--microphone", action="store_true", help="Record from the default microphone.")
    input_group.add_argument("--audio-file", help="Path to a wav, mp3, m4a, flac, or ogg file.")
    parser.add_argument("--mode", choices=["smart", "benchmark"], default="smart", help="STT routing mode.")
    parser.add_argument("--answer-mode", choices=["short", "detailed"], default="short", help="Answer style.")
    parser.add_argument("--provider", choices=["openai", "mock"], default="openai", help="Provider set to use.")
    parser.add_argument("--storage", choices=["postgres", "json"], default="postgres", help="Session storage backend.")
    parser.add_argument("--record-seconds", type=int, default=None, help="Microphone recording duration.")
    parser.add_argument("--output-dir", default="logs", help="Directory for JSON session logs.")
    parser.add_argument("--dotenv-path", default=".env", help="Path to the environment file.")
    return parser


def build_pipeline(config: AppConfig, *, provider: str, storage_backend: str, output_dir: str) -> InterviewPipeline:
    storage = PostgresSessionStorage(config.database_url) if storage_backend == "postgres" else JsonSessionStorage(output_dir)
    microphone_input = MicrophoneInputLayer(sample_rate=config.mic_sample_rate, channels=config.mic_channels)

    if provider == "mock":
        return InterviewPipeline(storage=storage, microphone_input=microphone_input)

    stt_router = STTRouter([OpenAISTTProvider(api_key=config.openai_api_key, model=config.openai_stt_model)])
    llm_router = LLMRouter([OpenAILLMProvider(api_key=config.openai_api_key, model=config.openai_llm_model)])
    return InterviewPipeline(
        stt_router=stt_router,
        llm_router=llm_router,
        storage=storage,
        microphone_input=microphone_input,
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = AppConfig.from_env(dotenv_path=args.dotenv_path)
        pipeline = build_pipeline(config, provider=args.provider, storage_backend=args.storage, output_dir=args.output_dir)
    except PipelineError as exc:
        print(f"Configuration error: {exc.user_message}")
        return 2

    if args.audio_file:
        result = pipeline.run_file(args.audio_file, mode=args.mode, answer_mode=args.answer_mode)
    else:
        record_seconds = args.record_seconds or config.mic_record_seconds
        print(f"Recording from microphone for {record_seconds} seconds...")
        result = pipeline.run_microphone(seconds=record_seconds, mode=args.mode, answer_mode=args.answer_mode)

    if result.transcript:
        print(f"Transcript ({result.selected_stt_model}, confidence {result.transcript.confidence:.2f}):")
        print(result.transcript.text)
        print()

    if result.answer:
        print(f"Answer ({result.selected_llm_model}):")
        print(result.answer.text)
        print()

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"- {error.stage}: {error.user_message}")
            provider_errors = getattr(error, "details", {}).get("provider_errors", [])
            for provider_error in provider_errors:
                print(f"  - {provider_error}")
        print()

    print(f"Latency: {result.latency_breakdown_ms}")
    print(f"Session log: {result.log_path}")
    return 1 if result.errors and not result.answer else 0


if __name__ == "__main__":
    raise SystemExit(main())
