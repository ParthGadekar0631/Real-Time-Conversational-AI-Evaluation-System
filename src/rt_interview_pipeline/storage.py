from __future__ import annotations

import json
from pathlib import Path

from .errors import PipelineError
from .models import PipelineResult


class JsonSessionStorage:
    def __init__(self, output_dir: str | Path = "logs") -> None:
        self.output_dir = Path(output_dir)

    def save(self, result: PipelineResult) -> str:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{result.session_id}.json"
            path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return str(path)
        except OSError as exc:
            raise PipelineError(
                "logging_storage",
                f"Could not write session log: {exc}",
                "Could not save the session log.",
                recoverable=False,
            ) from exc
