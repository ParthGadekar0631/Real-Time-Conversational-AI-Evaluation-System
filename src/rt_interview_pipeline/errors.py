from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorRecord:
    stage: str
    message: str
    user_message: str
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class PipelineError(Exception):
    def __init__(
        self,
        stage: str,
        message: str,
        user_message: str | None = None,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.user_message = user_message or message
        self.recoverable = recoverable
        self.details = details or {}

    def to_record(self) -> ErrorRecord:
        return ErrorRecord(
            stage=self.stage,
            message=self.message,
            user_message=self.user_message,
            recoverable=self.recoverable,
            details=self.details,
        )
