from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .cli import build_pipeline
from .config import AppConfig


class InterviewPipelineApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Real-Time Interview Response Pipeline")
        self.geometry("900x650")
        self.result_queue: queue.Queue = queue.Queue()

        self.record_seconds = tk.IntVar(value=8)
        self.answer_mode = tk.StringVar(value="short")
        self.storage = tk.StringVar(value="postgres")
        self.provider = tk.StringVar(value="openai")

        self._build_layout()
        self.after(200, self._poll_results)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(controls, text="Seconds").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=2, to=60, textvariable=self.record_seconds, width=6).pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(controls, text="Answer").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.answer_mode, values=["short", "detailed"], width=10, state="readonly").pack(
            side=tk.LEFT, padx=(6, 16)
        )

        ttk.Label(controls, text="Provider").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.provider, values=["openai", "mock"], width=10, state="readonly").pack(
            side=tk.LEFT, padx=(6, 16)
        )

        ttk.Label(controls, text="Storage").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.storage, values=["postgres", "json"], width=10, state="readonly").pack(
            side=tk.LEFT, padx=(6, 16)
        )

        self.record_button = ttk.Button(controls, text="Record", command=self._start_recording)
        self.record_button.pack(side=tk.RIGHT)

        self.status = ttk.Label(root, text="Ready")
        self.status.pack(fill=tk.X, pady=(0, 8))

        self.output = tk.Text(root, wrap=tk.WORD, height=30)
        self.output.pack(fill=tk.BOTH, expand=True)

    def _start_recording(self) -> None:
        self.record_button.configure(state=tk.DISABLED)
        self.status.configure(text=f"Recording for {self.record_seconds.get()} seconds...")
        self.output.delete("1.0", tk.END)
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self) -> None:
        try:
            config = AppConfig.from_env()
            pipeline = build_pipeline(
                config,
                provider=self.provider.get(),
                storage_backend=self.storage.get(),
                output_dir="logs",
            )
            result = pipeline.run_microphone(seconds=self.record_seconds.get(), answer_mode=self.answer_mode.get())
            self.result_queue.put(("result", result))
        except Exception as exc:
            self.result_queue.put(("error", exc))

    def _poll_results(self) -> None:
        try:
            kind, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_results)
            return

        self.record_button.configure(state=tk.NORMAL)
        if kind == "error":
            self.status.configure(text="Failed")
            messagebox.showerror("Pipeline Error", str(payload))
        else:
            self._render_result(payload)
        self.after(200, self._poll_results)

    def _render_result(self, result) -> None:
        self.status.configure(text=f"Done. Session: {result.session_id}")
        lines = []
        if result.transcript:
            lines.extend(
                [
                    f"Transcript model: {result.selected_stt_model}",
                    f"Confidence: {result.transcript.confidence:.2f}",
                    "",
                    result.transcript.text,
                    "",
                ]
            )
        if result.answer:
            lines.extend(["AI model:", result.selected_llm_model or "", "", "Answer:", result.answer.text, ""])
        if result.errors:
            lines.append("Errors:")
            lines.extend(f"- {error.stage}: {error.user_message}" for error in result.errors)
            lines.append("")
        lines.extend([f"Latency: {result.latency_breakdown_ms}", f"Saved: {result.log_path}"])
        self.output.insert(tk.END, "\n".join(lines))


def main() -> None:
    app = InterviewPipelineApp()
    app.mainloop()


if __name__ == "__main__":
    main()
