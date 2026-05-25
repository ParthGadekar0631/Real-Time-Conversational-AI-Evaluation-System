from __future__ import annotations

from .errors import PipelineError
from .models import AudioChunk, AudioQualityProfile


class AudioAnalysisModule:
    def analyze(self, chunks: list[AudioChunk]) -> AudioQualityProfile:
        if not chunks:
            raise PipelineError("audio_analysis", "No audio chunks to analyze", "No speech was detected.")

        data = b"".join(chunk.data for chunk in chunks)
        if len(data) < 8:
            raise PipelineError("audio_analysis", "Audio is too short", "The audio is too short to analyze.")

        byte_values = list(data)
        zero_count = sum(1 for value in byte_values if value in (0, 32))
        silence_ratio = min(1.0, zero_count / len(byte_values))

        spread = max(byte_values) - min(byte_values)
        volume_db = round(-60.0 + (spread / 255.0) * 60.0, 2)
        average = sum(byte_values) / len(byte_values)
        noise = sum(abs(value - average) for value in byte_values) / len(byte_values)
        noise_snr = round(max(0.0, min(40.0, 40.0 - noise / 2.0)), 2)

        speech_duration_ms = sum(chunk.duration_ms for chunk in chunks)
        clarity_score = round(max(0.0, min(1.0, (1.0 - silence_ratio) * 0.65 + (noise_snr / 40.0) * 0.35)), 3)

        if clarity_score < 0.05:
            raise PipelineError("audio_analysis", "No usable speech detected", "No usable speech was detected.")

        return AudioQualityProfile(
            volume_db=volume_db,
            noise_snr=noise_snr,
            silence_ratio=round(silence_ratio, 3),
            speech_duration_ms=speech_duration_ms,
            clarity_score=clarity_score,
        )
