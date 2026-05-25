from __future__ import annotations

from pathlib import Path

from .errors import PipelineError
from .models import AudioChunk

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class AudioInputLayer:
    def __init__(self, *, chunk_size_bytes: int = 32_000, chunk_duration_ms: int = 1_000) -> None:
        self.chunk_size_bytes = chunk_size_bytes
        self.chunk_duration_ms = chunk_duration_ms

    def load_file(self, path: str | Path) -> list[AudioChunk]:
        audio_path = Path(path)
        if not audio_path.exists():
            raise PipelineError(
                "audio_input",
                f"Audio file not found: {audio_path}",
                "The selected audio file could not be found.",
                recoverable=False,
            )

        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            raise PipelineError(
                "audio_input",
                f"Unsupported audio format: {audio_path.suffix}",
                "Use a supported format: wav, mp3, m4a, flac, or ogg.",
            )

        data = audio_path.read_bytes()
        if not data:
            raise PipelineError("audio_input", "Empty audio file", "The audio file is empty.")

        return self._chunk(data)

    def _chunk(self, data: bytes) -> list[AudioChunk]:
        return [
            AudioChunk(index=index, data=data[start : start + self.chunk_size_bytes], duration_ms=self.chunk_duration_ms)
            for index, start in enumerate(range(0, len(data), self.chunk_size_bytes))
        ]
