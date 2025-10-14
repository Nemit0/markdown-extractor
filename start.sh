#!/bin/bash
# Start script for Markdown Extractor web app

echo "Starting Markdown Extractor..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
