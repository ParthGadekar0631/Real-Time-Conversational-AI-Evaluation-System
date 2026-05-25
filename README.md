# Adaptive Real-Time AI Interview Response Pipeline

This repository contains a Python implementation for an adaptive interview-response pipeline. It records microphone audio, evaluates audio quality, routes speech-to-text work through OpenAI or mock providers, ranks transcripts, routes the final transcript to an LLM, post-processes the answer, and stores a structured session log in PostgreSQL or local JSON.

![Adaptive real-time AI interview response pipeline](docs/assets/adaptive-real-time-ai-interview-response-pipeline.png)

## Pipeline

```mermaid
flowchart TD
    A["Audio Input Layer<br/>Microphone or audio file"] --> B["Audio Capture<br/>Buffering and chunking"]
    B --> C["Audio Analysis Module<br/>Volume, noise, silence, speech duration, clarity"]
    C --> D["STT Router<br/>Smart routing or benchmark mode"]
    D --> E["Transcript Evaluation Engine<br/>Score, rank, select best transcript"]
    E --> F{"Valid transcript?"}
    F -- "No" --> EH["Global Error Handling<br/>retry, fallback, log, user message"]
    EH --> D
    F -- "Yes" --> G["LLM Router<br/>classify question and select model"]
    G --> H["Answer Generation Module<br/>format, limit length, readability"]
    H --> I["Output Display<br/>transcript, answer, models, latency, confidence"]
    I --> J["Logging and Storage<br/>PostgreSQL or JSON session log"]
```

## Current Implementation

- Live microphone recording.
- Audio file ingestion with validation and chunking for testing.
- Heuristic audio quality profiling.
- OpenAI `gpt-4o-transcribe` provider.
- OpenAI GPT response provider.
- STT router with smart-routing and benchmark modes.
- Transcript scoring based on confidence, completeness, latency, and cost.
- LLM router with basic question classification.
- Answer generation and formatting.
- PostgreSQL session storage.
- JSON session logging for local fallback/debugging.
- CLI and Tkinter desktop GUI.
- Mock providers so the system runs without external API keys.

## Run Locally

First, revoke any API key that was shared in screenshots or chat, then create a fresh key. Put the fresh key in a local `.env` file. Do not commit `.env`.

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum `.env` values for OpenAI plus PostgreSQL:

```env
OPENAI_API_KEY=your_new_rotated_key_here
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_LLM_MODEL=gpt-4.1-mini
DATABASE_URL=postgresql://interview_user:interview_password@localhost:5432/interview_pipeline
MIC_RECORD_SECONDS=8
```

Start Docker PostgreSQL:

```powershell
docker compose up -d postgres
```

Install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Run the live microphone CLI with OpenAI and PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m rt_interview_pipeline.cli --microphone --provider openai --storage postgres --record-seconds 8
```

Run the desktop GUI:

```powershell
.\.venv\Scripts\python.exe -m rt_interview_pipeline.gui
```

Run without API keys using mock providers and JSON logs:

```powershell
.\.venv\Scripts\python.exe -m rt_interview_pipeline.cli --microphone --provider mock --storage json --record-seconds 5
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Remaining Local Requirements

To run the full OpenAI + microphone + PostgreSQL path, provide:

- A fresh rotated `OPENAI_API_KEY` in `.env`.
- A running PostgreSQL database.
- A valid `DATABASE_URL` in `.env`.
- Windows microphone permission for the terminal or Python process.

## Architecture

The detailed textual architecture is in [docs/architecture.md](docs/architecture.md).
