from __future__ import annotations

import json

from .errors import PipelineError
from .models import PipelineResult


class PostgresSessionStorage:
    def __init__(self, database_url: str | None) -> None:
        if not database_url:
            raise PipelineError(
                "logging_storage",
                "DATABASE_URL is not configured",
                "Set DATABASE_URL in your .env file before using PostgreSQL storage.",
                recoverable=False,
            )
        self.database_url = database_url

    def save(self, result: PipelineResult) -> str:
        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise PipelineError(
                "logging_storage",
                "PostgreSQL dependency is not installed",
                "Install project dependencies before using PostgreSQL storage.",
                recoverable=False,
            ) from exc

        payload = result.to_dict()
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS interview_sessions (
                            session_id TEXT PRIMARY KEY,
                            created_at TIMESTAMPTZ NOT NULL,
                            selected_stt_model TEXT,
                            selected_llm_model TEXT,
                            transcript TEXT,
                            answer TEXT,
                            latency_breakdown_ms JSONB NOT NULL,
                            payload JSONB NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        INSERT INTO interview_sessions (
                            session_id,
                            created_at,
                            selected_stt_model,
                            selected_llm_model,
                            transcript,
                            answer,
                            latency_breakdown_ms,
                            payload
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            selected_stt_model = EXCLUDED.selected_stt_model,
                            selected_llm_model = EXCLUDED.selected_llm_model,
                            transcript = EXCLUDED.transcript,
                            answer = EXCLUDED.answer,
                            latency_breakdown_ms = EXCLUDED.latency_breakdown_ms,
                            payload = EXCLUDED.payload
                        """,
                        (
                            result.session_id,
                            result.created_at,
                            result.selected_stt_model,
                            result.selected_llm_model,
                            result.transcript.text if result.transcript else None,
                            result.answer.text if result.answer else None,
                            Jsonb(result.latency_breakdown_ms),
                            Jsonb(payload),
                        ),
                    )
            return f"postgresql:interview_sessions/{result.session_id}"
        except Exception as exc:
            details = {"payload_preview": json.dumps(payload)[:500]}
            raise PipelineError(
                "logging_storage",
                f"Could not save session to PostgreSQL: {exc}",
                "Could not save the session to PostgreSQL. Check DATABASE_URL and database availability.",
                recoverable=False,
                details=details,
            ) from exc
