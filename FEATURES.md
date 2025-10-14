# Markdown Extractor - Feature Summary

## Implemented Features

### 1. Real-time Progress Bar
- Visual progress bar showing percentage complete
- Live updates during processing via WebSocket
- Shows current file and page being processed
- Example: "Processing page 5 (File 2/3 | Page 5/10)"

### 2. Auto-OCR Detection Toggle
- Smart OCR detection feature
- Randomly samples 3-5 pages from each PDF
- Compares text extraction vs image quality
- Automatically enables OCR if >50% of samples need it
- Saves API costs by skipping OCR when not needed

### 3. Multiple File Support
- Upload multiple PDFs and images at once
- Drag-and-drop interface for easy file selection
- Processes each file sequentially with progress tracking
- All results packaged into single ZIP download

### 4. WebSocket Real-time Updates
- Live progress updates during processing
- No polling required - push-based updates
- Connection per job ID
- Graceful fallback if WebSocket fails

## How It Works

### Auto-OCR Detection
1. User enables "Auto-detect OCR requirement" checkbox
2. For each PDF, backend samples 3-5 random pages
3. Extracts text using markitdown from each sample
4. If more than half have <50 characters, enables OCR for all pages
5. Saves time and API costs for text-based PDFs

### Progress Tracking
1. Frontend connects to WebSocket endpoint `/ws/{job_id}`
2. Backend sends progress updates:
   - `file_start`: Starting a new file
   - `progress`: Page-level progress with current/total
   - `file_complete`: File finished
   - `file_error`: Error occurred
3. Progress bar updates in real-time
4. Details show current status and file/page counts

### Multiple Files
1. User selects/drops multiple files
2. Files displayed in list with size and remove option
3. All files uploaded together
4. Processed sequentially with individual progress
5. Results combined in ZIP with folder per file

## API Endpoints

- `GET /` - Web interface
- `POST /api/upload` - Upload multiple files
- `POST /api/process/{job_id}?auto_ocr=true` - Process with optional auto-OCR
- `GET /api/download/{job_id}` - Download ZIP results
- `WS /ws/{job_id}` - WebSocket for progress updates

## Usage Example

```bash
# Start server
cd markdown-extractor
python main.py

# Open browser
open http://localhost:8000

# Or with curl (API)
curl -X POST -F "files=@doc1.pdf" -F "files=@doc2.pdf" http://localhost:8000/api/upload
curl -X POST http://localhost:8000/api/process/{job_id}?auto_ocr=true
curl -O http://localhost:8000/api/download/{job_id}
```

## Output Structure

```
{job_id}.zip
├── doc1/
│   ├── page-001.md
│   ├── page-001.png
│   ├── page-002.md
│   ├── page-002.png
│   └── document_refined.md
└── doc2/
    ├── page-001.md
    ├── page-001.png
    └── document_refined.md
```

## Configuration

Edit these in `converter.py`:
- `model`: GPT model (default: "gpt-4o")
- `dpi`: PDF render resolution (default: 220)
- `max_concurrency`: Parallel OpenAI calls (default: 5)
- `min_text_length`: OCR threshold (default: 50 chars)

## Future Enhancements

Possible improvements:
- Pause/resume processing
- Custom OCR languages
- Batch job queue
- Results preview before download
- Token usage estimation before processing
