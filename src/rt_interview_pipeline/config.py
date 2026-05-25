from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = True) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_stt_model: str
    openai_llm_model: str
    database_url: str | None
    mic_sample_rate: int
    mic_channels: int
    mic_record_seconds: int

    @classmethod
    def from_env(cls, *, dotenv_path: str | Path = ".env") -> "AppConfig":
        load_dotenv(dotenv_path)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_stt_model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe"),
            openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini"),
            database_url=os.getenv("DATABASE_URL") or None,
            mic_sample_rate=int(os.getenv("MIC_SAMPLE_RATE", "16000")),
            mic_channels=int(os.getenv("MIC_CHANNELS", "1")),
            mic_record_seconds=int(os.getenv("MIC_RECORD_SECONDS", "8")),
        )
