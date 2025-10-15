# Markdown Extractor Web App

A web application for extracting clean markdown from PDFs, Office files, and images with automatic OCR support for Korean and English text.

## Features

- **Drag-and-drop interface** for multiple file uploads
- **Supports** PDF, PNG, JPG, JPEG, DOCX, PPTX, XLSX files
- **Multiple AI models** - OpenAI GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and more
- **Automatic OCR** for image-based documents (Korean + English)
- **AI vision models** for markdown cleanup and formatting
- **Real-time progress bar** with WebSocket updates
- **Auto-OCR detection** - samples 3-5 random pages to determine if OCR is needed
- **Multiple files** - process many files at once, download as ZIP
- **Image extraction** - automatically extracts and links images from PDFs
- **Concurrent processing** for multiple pages (5 workers)
- **Smart caching** - skip already processed pages
- **Live updates** - see current file and page being processed

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your API keys for the AI model(s) you want to use:
     - `OPENAI_API_KEY` for OpenAI models (GPT-4o, GPT-4o Mini, etc.)
     - `ANTHROPIC_API_KEY` for Claude models (Claude 3.5 Sonnet, etc.)
     - `GOOGLE_API_KEY` for Gemini models (Gemini 1.5 Pro, etc.)

3. Run the application:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Open browser to `http://localhost:8000`
2. Select an AI model from the dropdown (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, etc.)
3. Drag and drop PDF, Office files, or image files
4. Optionally enable "Auto-OCR detection" for automatic OCR requirement detection
5. Click "Process Files"
6. Watch the real-time progress bar
7. Download the ZIP file with all extracted markdown files and images

## API Endpoints

- `GET /` - Web interface
- `POST /api/upload` - Upload files
- `POST /api/process/{job_id}` - Process uploaded files
- `GET /api/download/{job_id}` - Download results as ZIP
- `DELETE /api/cleanup/{job_id}` - Clean up job files
- `GET /api/health` - Health check

## Output Format

Each processed file will have:
- Individual page markdown files (`page-001.md`, `page-002.md`, etc.)
- Page images as PNG files
- Extracted images from PDFs in `images/` directory
- Combined markdown file (`document_refined.md`) with all pages separated by `#### Page N` headers and image references

## Supported AI Models

### OpenAI
- GPT-4o (recommended)
- GPT-4o Mini
- GPT-4 Turbo

### Anthropic
- Claude 3.5 Sonnet (recommended)
- Claude 3 Opus
- Claude 3 Haiku

### Google
- Gemini 1.5 Pro
- Gemini 1.5 Flash

## Configuration

Edit these parameters in `converter.py`:

- `model`: AI model to use (can be selected in UI)
- `dpi`: Image resolution for PDF rendering (default: 220)
- `temperature`: AI temperature (default: 0.0)
- `max_concurrency`: Number of concurrent AI calls (default: 5)
- `min_text_length`: Minimum text length to skip OCR (default: 50)

## Requirements

- Python 3.8+
- At least one AI provider API key (OpenAI, Anthropic, or Google)
- Internet connection for AI API calls
