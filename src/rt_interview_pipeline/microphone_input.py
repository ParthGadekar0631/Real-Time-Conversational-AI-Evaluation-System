from __future__ import annotations

import io
import wave

from .errors import PipelineError
from .models import AudioChunk


class MicrophoneInputLayer:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_size_bytes: int = 32_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size_bytes = chunk_size_bytes

    def record(self, *, seconds: int) -> list[AudioChunk]:
        if seconds <= 0:
            raise PipelineError("audio_input", "Recording duration must be positive", "Recording duration must be positive.")

        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise PipelineError(
                "audio_input",
                "Microphone dependencies are not installed",
                "Install project dependencies before using microphone input.",
                recoverable=False,
                details={"missing_dependency": str(exc)},
            ) from exc

        try:
            frames = sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()
        except Exception as exc:
            raise PipelineError(
                "audio_input",
                f"Microphone recording failed: {exc}",
                "Could not record from the microphone. Check microphone permissions and device settings.",
            ) from exc

        if frames.size == 0:
            raise PipelineError("audio_input", "Empty microphone recording", "No audio was recorded.")

        wav_bytes = self._to_wav_bytes(frames.astype(np.int16).tobytes())
        return self._chunk(wav_bytes, seconds)

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_bytes)
        return output.getvalue()

    def _chunk(self, data: bytes, seconds: int) -> list[AudioChunk]:
        chunk_count = max(1, (len(data) + self.chunk_size_bytes - 1) // self.chunk_size_bytes)
        duration_ms = max(1, int(seconds * 1000 / chunk_count))
        return [
            AudioChunk(index=index, data=data[start : start + self.chunk_size_bytes], duration_ms=duration_ms)
            for index, start in enumerate(range(0, len(data), self.chunk_size_bytes))
        ]
