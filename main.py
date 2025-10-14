
import os
import uuid
import shutil
import zipfile
import asyncio
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from converter import process_file

load_dotenv()

app = FastAPI(title="Markdown Extractor", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_progress(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json(message)
            except:
                self.disconnect(job_id)

manager = ConnectionManager()


@app.get("/")
async def root():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return {"message": "Markdown Extractor API", "docs": "/docs"}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id

    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Only PDF, PNG, JPG, JPEG are supported."
            )

        file_path = job_upload_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        uploaded_files.append({
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content)
        })

    return {
        "job_id": job_id,
        "files": uploaded_files,
        "message": f"Uploaded {len(uploaded_files)} file(s). Use /api/process/{job_id} to process."
    }


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id)


@app.post("/api/process/{job_id}")
async def process_job(job_id: str, auto_ocr: bool = False):

    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id

    if not job_upload_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    files = list(job_upload_dir.glob("*"))

    if not files:
        raise HTTPException(status_code=400, detail="No files to process")

    results = []

    async def progress_callback(status: str, current: int, total: int):
        await manager.send_progress(job_id, {
            "type": "progress",
            "filename": file_path.name,
            "status": status,
            "current": current,
            "total": total,
            "file_index": len(results) + 1,
            "total_files": len(files)
        })

    for file_idx, file_path in enumerate(files, 1):
        try:
            await manager.send_progress(job_id, {
                "type": "file_start",
                "filename": file_path.name,
                "file_index": file_idx,
                "total_files": len(files)
            })

            file_output_dir = job_output_dir / file_path.stem

            import concurrent.futures
            loop = asyncio.get_event_loop()

            def sync_progress(status, current, total):
                asyncio.run_coroutine_threadsafe(
                    progress_callback(status, current, total),
                    loop
                )

            result = await loop.run_in_executor(
                None,
                lambda: process_file(
                    input_path=file_path,
                    out_dir=file_output_dir,
                    model="gpt-4o",
                    dpi=220,
                    temperature=0.0,
                    max_concurrency=5,
                    min_text_length=50,
                    use_ocr_if_needed=True,
                    skip_openai=False,
                    auto_ocr_detect=auto_ocr,
                    progress_callback=sync_progress,
                )
            )

            results.append({
                "filename": file_path.name,
                "success": True,
                "pages_processed": result["pages_processed"],
                "total_pages": result["total_pages"],
                "token_usage": result["token_usage"],
                "output_dir": str(file_output_dir),
            })

            await manager.send_progress(job_id, {
                "type": "file_complete",
                "filename": file_path.name,
                "file_index": file_idx,
                "total_files": len(files)
            })

        except Exception as e:
            results.append({
                "filename": file_path.name,
                "success": False,
                "error": str(e),
            })

            await manager.send_progress(job_id, {
                "type": "file_error",
                "filename": file_path.name,
                "error": str(e),
                "file_index": file_idx,
                "total_files": len(files)
            })

    return {
        "job_id": job_id,
        "results": results,
        "download_url": f"/api/download/{job_id}"
    }


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    job_output_dir = OUTPUT_DIR / job_id

    if not job_output_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found or not processed yet")

    zip_path = OUTPUT_DIR / f"{job_id}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(job_output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(job_output_dir)
                zipf.write(file_path, arcname)

    return FileResponse(
        path=str(zip_path),
        filename=f"markdown_extraction_{job_id}.zip",
        media_type="application/zip"
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    zip_path = OUTPUT_DIR / f"{job_id}.zip"

    cleaned = []

    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)
        cleaned.append("uploads")

    if job_output_dir.exists():
        shutil.rmtree(job_output_dir)
        cleaned.append("outputs")

    if zip_path.exists():
        zip_path.unlink()
        cleaned.append("zip")

    return {
        "job_id": job_id,
        "cleaned": cleaned,
        "message": "Cleanup completed"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "markdown-extractor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
