from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rt_interview_pipeline.pipeline import InterviewPipeline
from rt_interview_pipeline.storage import JsonSessionStorage


class InterviewPipelineTests(unittest.TestCase):
    def test_pipeline_runs_with_mock_providers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_file = root / "sample.wav"
            audio_file.write_text("Explain the tradeoffs of WebSockets for interview systems.", encoding="utf-8")

            pipeline = InterviewPipeline(storage=JsonSessionStorage(root / "logs"))
            result = pipeline.run_file(audio_file)

            self.assertIsNotNone(result.audio_quality)
            self.assertIsNotNone(result.transcript)
            self.assertIsNotNone(result.answer)
            self.assertTrue(result.selected_stt_model)
            self.assertTrue(result.selected_llm_model)
            self.assertTrue(result.log_path)
            self.assertTrue(Path(result.log_path).exists())
            self.assertEqual(result.errors, [])

    def test_invalid_audio_format_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_file = root / "sample.txt"
            audio_file.write_text("not an audio extension", encoding="utf-8")

            pipeline = InterviewPipeline(storage=JsonSessionStorage(root / "logs"))
            result = pipeline.run_file(audio_file)

            self.assertIsNone(result.answer)
            self.assertEqual(result.errors[0].stage, "audio_input")
            self.assertTrue(result.log_path)


if __name__ == "__main__":
    unittest.main()
