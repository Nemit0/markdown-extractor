# Markdown Extractor Web App

A web application for extracting clean markdown from PDFs and images with automatic OCR support for Korean and English text.

## Features

- **Drag-and-drop interface** for multiple file uploads
- **Supports** PDF, PNG, JPG, JPEG files
- **Automatic OCR** for image-based documents (Korean + English)
- **GPT-4o vision** model for markdown cleanup and formatting
- **Real-time progress bar** with WebSocket updates
- **Auto-OCR detection** - samples 3-5 random pages to determine if OCR is needed
- **Multiple files** - process many files at once, download as ZIP
- **Concurrent processing** for multiple pages (5 workers)
- **Smart caching** - skip already processed pages
- **Live updates** - see current file and page being processed

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env` file with your `OPENAI_API_KEY`

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
2. Drag and drop PDF or image files
3. Click "Process Files"
4. Wait for processing to complete
5. Download the ZIP file with all extracted markdown files

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
- Combined markdown file (`document_refined.md`) with all pages separated by `#### Page N` headers

## Configuration

Edit these parameters in `converter.py`:

- `model`: GPT model to use (default: "gpt-4o")
- `dpi`: Image resolution for PDF rendering (default: 220)
- `temperature`: OpenAI temperature (default: 0.0)
- `max_concurrency`: Number of concurrent OpenAI calls (default: 5)
- `min_text_length`: Minimum text length to skip OCR (default: 50)

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for OpenAI API calls
