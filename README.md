# Meeting Minutes Generator

## Overview

Meeting Minutes Generator is a Python-based application that transcribes audio recordings of meetings and generates structured, professional meeting minutes using advanced AI models.

It supports two operational modes:

- **Lite Mode**: Uses OpenAI's GPT-3.5 Turbo for fast and efficient processing.
- **Pro Mode**: Uses Hugging Face's GPT-OSS-20B model with 4-bit quantization for higher-quality output, optimized for GPU-enabled systems.

The application uses OpenAI’s Whisper for transcription, regular expressions for structured data extraction, and is containerized with Docker for consistent deployment across platforms like Railway.

---

## Features

- **Audio Transcription**  
  Converts meeting audio files into text using OpenAI's Whisper model (`whisper-1`). Supports various audio formats and performs well across different accents and noise levels.

- **Structured Data Extraction**  
  Extracts key information such as emails, dates, times, and action phrases using regular expressions. Outputs structured data as JSON.

- **Dual-Mode Minutes Generation**
  - **Lite Mode**: Uses OpenAI's GPT-3.5 Turbo for rapid generation of meeting minutes, suitable for resource-constrained environments.
  - **Pro Mode**: Employs Hugging Face's GPT-OSS-20B with 4-bit quantization for higher-quality results.

- **Consistent Output Format**  
  Generates markdown-formatted minutes with sections for Meeting Overview, Key Decisions, Discussion Highlights, Action Items (table), Key Metrics, and Follow-up Required.

- **Dockerized Deployment**  
  Packaged in a lightweight Docker container using Python 3.10 slim, with CPU-based PyTorch for compatibility with platforms like Railway.

- **Error Handling and Logging**  
  Comprehensive error handling and logging are included for easier debugging and user feedback.

---

## Technical Architecture

### Core Components

The application is implemented in a single `app.py` file with a modular `MeetingMinutesGenerator` class encapsulating the following functionality:

#### OpenAI Integration

- **Whisper Transcription**  
  The `transcribe_audio` method uses OpenAI's Whisper model (`whisper-1`) to convert audio files into text. Supports binary audio input and requires a valid OpenAI API key.

- **GPT-3.5 Turbo (Lite Mode)**  
  The `generate_minutes_lite` method sends transcriptions (truncated to 4,000 characters) to OpenAI's Chat Completion API (`gpt-3.5-turbo`) using a structured prompt, generating markdown-formatted minutes with a limit of 1,500 tokens.

#### Hugging Face Integration (Pro Mode)

- **Model Setup**  
  The `setup_huggingface` method authenticates using a Hugging Face token and loads the GPT-OSS-20B model and tokenizer using `AutoModelForCausalLM` and `AutoTokenizer`.

- **Quantization**  
  Uses `bitsandbytes` and `BitsAndBytesConfig` for 4-bit quantization:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_use_double_quant=True`
  - `bnb_4bit_compute_dtype=torch.bfloat16`

  This reduces memory usage from approximately 40GB to 5–10GB. Falls back to full precision (`torch.float32`) on CPU.

- **Generation**  
  The `generate_minutes_pro` method processes transcriptions (truncated to 3,000 characters) using `apply_chat_template` and generates output with `model.generate()` using:
  - `max_new_tokens=1500`
  - `temperature=0.7`
  - `repetition_penalty=1.1`

#### Structured Data Extraction

The `extract_structured_data` method uses regular expressions to extract:
- Emails
- Dates
- Times
- Action phrases

Each category is limited to 5 results to manage output size.

#### Prompt Engineering

Both modes use identical prompts for consistent output:
- **System Prompt**: Defines the assistant as an expert meeting secretary.
- **User Prompt**: Provides a structured markdown template with placeholders.

---

## Dependencies

Listed in `requirements.txt`:

- `torch==2.5.1+cpu`, `torchvision`, `torchaudio`: CPU-based PyTorch
- `transformers==4.48.3`: For loading Hugging Face models
- `accelerate==1.3.0`: For model device placement (`device_map="auto"`)
- `bitsandbytes==0.46.0`: Enables 4-bit quantization
- `huggingface-hub`: Handles Hugging Face login and downloads
- `openai`: For Whisper and GPT-3.5 Turbo APIs
- `requests`: For HTTP communication (e.g., health checks)
- `python-dotenv`: Loads environment variables from `.env`
- `gunicorn`: Optional production web server support

---

## Docker Configuration

The Dockerfile:

- Based on `python:3.10-slim`
- Installs system dependencies: `git`, `wget`, `curl`, `build-essential`
- Uses CPU-based PyTorch
- Creates a non-root user (`appuser`) for security
- Exposes a dynamic `PORT` environment variable
- Adds a basic health check: `curl http://localhost:$PORT/`

---

## Installation

### Prerequisites

- Python 3.10 or later
- OpenAI API key (required for transcription and Lite Mode)
- Hugging Face token (required for Pro Mode)
- Docker (optional, for container deployment)
- CUDA-compatible GPU (optional, for Pro Mode quantization)

### Steps

1. **Clone the Repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
