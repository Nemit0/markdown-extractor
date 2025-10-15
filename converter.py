
from __future__ import annotations

import base64
import io
import tempfile
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from markitdown import MarkItDown
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdf import PdfReader, PdfWriter
from PIL import Image
from openai import OpenAI
import anthropic
import google.generativeai as genai
import os


_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return _ocr_reader


def ocr_image(image_path: Path) -> str:
    reader = get_ocr_reader()
    result = reader.readtext(str(image_path), detail=0)
    return "\n".join(result)


def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def extract_page_pdf_bytes(src_pdf: Path, page_index: int) -> bytes:
    reader = PdfReader(str(src_pdf))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    bio = io.BytesIO()
    writer.write(bio)
    bio.seek(0)
    return bio.getvalue()


def markitdown_convert_page(md: MarkItDown, page_pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(page_pdf_bytes)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        result = md.convert(str(tmp_path))
        return (result.text_content or "").strip()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def render_page_png(pdf_doc: pdfium.PdfDocument, page_index: int, out_png: Path, dpi: int = 220) -> None:
    scale = float(dpi) / 72.0
    page = pdf_doc[page_index]
    pil = page.render(scale=scale).to_pil()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_png, format="PNG")


def extract_images_from_page(pdf_doc: pdfium.PdfDocument, page_index: int, out_dir: Path, page_no: int) -> List[str]:
    page = pdf_doc[page_index]
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    try:
        obj_searcher = page.get_objects(
            filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,),
            max_depth=2
        )
        image_objects = list(obj_searcher)

        for img_idx, image_obj in enumerate(image_objects, 1):
            try:
                image_filename = f"page-{page_no:03d}_image-{img_idx}.png"
                image_path = images_dir / image_filename

                pil_image = image_obj.get_bitmap(render=True).to_pil()
                pil_image.save(str(image_path), format="PNG")

                extracted_images.append(f"images/{image_filename}")
            except Exception as e:
                continue

    except Exception as e:
        pass

    return extracted_images


def build_messages(raw_markdown: str, page_png_path: Path, page_no: int, extracted_images: List[str] = None) -> list[dict[str, Any]]:
    img_b64 = b64_image(page_png_path)

    image_info = ""
    if extracted_images and len(extracted_images) > 0:
        image_info = f"\n\nExtracted images from this page (reference these in markdown):\n"
        for img_path in extracted_images:
            image_info += f"- {img_path}\n"
        image_info += "\nUse ![alt text]({image_path}) syntax to reference these images in appropriate locations in the markdown."

    return [
        {
            "role": "system",
            "content": (
                "You are a precise Markdown formatter. Given raw text and a page image, "
                "reconstruct clean, semantic Markdown for this single page only. "
                "Preserve all content; do not add text not present on this page. "
                "If the markdown text format does not match the image, prioritize the document structure from the image. "
                "Infer headings, lists, tables (GitHub Flavored Markdown). "
                "Keep math as LaTeX ($...$ / $$...$$). "
                "If images were extracted from the page, you will be provided with their paths. "
                "Reference these extracted images in the markdown using ![description](image_path) syntax where appropriate. "
                "Place image references where they logically appear in the document flow. "
                "Return ONLY the Markdown for this page - do NOT wrap it in code blocks or markdown fences. "
                "Preserve Chart formatting. If the text does not respect the chart from the image, reformat it to match the chart."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Page {page_no} — raw markdown to clean:{image_info}\n\n{raw_markdown}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
            ],
        },
    ]


def ai_cleanup_page(
    model: str,
    page_no: int,
    raw_markdown: str,
    png_path: Path,
    temperature: float,
    extracted_images: List[str] = None,
) -> Tuple[int, str, Dict[str, int]]:

    if model.startswith("claude"):
        return claude_cleanup_page(model, page_no, raw_markdown, png_path, temperature, extracted_images)
    elif model.startswith("gemini"):
        return gemini_cleanup_page(model, page_no, raw_markdown, png_path, temperature, extracted_images)
    else:
        return openai_cleanup_page(model, page_no, raw_markdown, png_path, temperature, extracted_images)


def openai_cleanup_page(
    model: str,
    page_no: int,
    raw_markdown: str,
    png_path: Path,
    temperature: float,
    extracted_images: List[str] = None,
) -> Tuple[int, str, Dict[str, int]]:
    client = OpenAI()
    messages = build_messages(raw_markdown, png_path, page_no, extracted_images)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    md_text = (resp.choices[0].message.content or "").strip()
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") else 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if hasattr(resp, "usage") else 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) if hasattr(resp, "usage") else 0,
    }
    return page_no, md_text, usage


def claude_cleanup_page(
    model: str,
    page_no: int,
    raw_markdown: str,
    png_path: Path,
    temperature: float,
    extracted_images: List[str] = None,
) -> Tuple[int, str, Dict[str, int]]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    img_b64 = b64_image(png_path)
    image_info = ""
    if extracted_images and len(extracted_images) > 0:
        image_info = f"\n\nExtracted images from this page (reference these in markdown):\n"
        for img_path in extracted_images:
            image_info += f"- {img_path}\n"
        image_info += "\nUse ![alt text]({image_path}) syntax to reference these images in appropriate locations in the markdown."

    system_prompt = (
        "You are a precise Markdown formatter. Given raw text and a page image, "
        "reconstruct clean, semantic Markdown for this single page only. "
        "Preserve all content; do not add text not present on this page. "
        "If the markdown text format does not match the image, prioritize the document structure from the image. "
        "Infer headings, lists, tables (GitHub Flavored Markdown). "
        "Keep math as LaTeX ($...$ / $$...$$). "
        "If images were extracted from the page, you will be provided with their paths. "
        "Reference these extracted images in the markdown using ![description](image_path) syntax where appropriate. "
        "Place image references where they logically appear in the document flow. "
        "Return ONLY the Markdown for this page - do NOT wrap it in code blocks or markdown fences. "
        "Preserve Chart formatting. If the text does not respect the chart from the image, reformat it to match the chart."
    )

    user_content = f"Page {page_no} — raw markdown to clean:{image_info}\n\n{raw_markdown}"

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": user_content
                }
            ],
        }]
    )

    md_text = message.content[0].text.strip()
    usage = {
        "prompt_tokens": message.usage.input_tokens,
        "completion_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
    }
    return page_no, md_text, usage


def gemini_cleanup_page(
    model: str,
    page_no: int,
    raw_markdown: str,
    png_path: Path,
    temperature: float,
    extracted_images: List[str] = None,
) -> Tuple[int, str, Dict[str, int]]:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    image_info = ""
    if extracted_images and len(extracted_images) > 0:
        image_info = f"\n\nExtracted images from this page (reference these in markdown):\n"
        for img_path in extracted_images:
            image_info += f"- {img_path}\n"
        image_info += "\nUse ![alt text]({image_path}) syntax to reference these images in appropriate locations in the markdown."

    system_prompt = (
        "You are a precise Markdown formatter. Given raw text and a page image, "
        "reconstruct clean, semantic Markdown for this single page only. "
        "Preserve all content; do not add text not present on this page. "
        "If the markdown text format does not match the image, prioritize the document structure from the image. "
        "Infer headings, lists, tables (GitHub Flavored Markdown). "
        "Keep math as LaTeX ($...$ / $$...$$). "
        "If images were extracted from the page, you will be provided with their paths. "
        "Reference these extracted images in the markdown using ![description](image_path) syntax where appropriate. "
        "Place image references where they logically appear in the document flow. "
        "Return ONLY the Markdown for this page - do NOT wrap it in code blocks or markdown fences. "
        "Preserve Chart formatting. If the text does not respect the chart from the image, reformat it to match the chart."
    )

    user_content = f"Page {page_no} — raw markdown to clean:{image_info}\n\n{raw_markdown}"

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={"temperature": temperature}
    )

    img = Image.open(png_path)

    response = model_instance.generate_content([
        system_prompt + "\n\n" + user_content,
        img
    ])

    md_text = response.text.strip()

    usage = {
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0,
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0,
        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) if hasattr(response, "usage_metadata") else 0,
    }
    return page_no, md_text, usage


def combine_pages(out_dir: Path, combined_name: str = "document_refined.md") -> Path:
    parts = []
    for md_file in sorted(out_dir.glob("page-*.md")):
        page_num = int(md_file.stem.split("-")[1])
        content = md_file.read_text(encoding="utf-8").rstrip()
        parts.append(f"#### Page {page_num}\n\n{content}\n")
    out_path = out_dir / combined_name
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ['.png', '.jpg', '.jpeg']


def check_if_ocr_needed(input_path: Path, sample_size: int = 5, min_text_length: int = 50) -> bool:
    if is_image_file(input_path):
        return True

    try:
        pdf_doc = pdfium.PdfDocument(str(input_path))
        total_pages = len(pdf_doc)

        num_samples = min(random.randint(3, 5), total_pages)
        sample_pages = random.sample(range(total_pages), num_samples)

        md = MarkItDown()
        low_text_count = 0

        for page_idx in sample_pages:
            page_pdf_bytes = extract_page_pdf_bytes(input_path, page_idx)
            raw_md = markitdown_convert_page(md, page_pdf_bytes)

            if len(raw_md.strip()) < min_text_length:
                low_text_count += 1

        needs_ocr = low_text_count > (num_samples / 2)
        return needs_ocr

    except Exception as e:
        return False


def process_file(
    input_path: Path,
    out_dir: Path,
    model: str = "gpt-4o",
    dpi: int = 220,
    temperature: float = 0.0,
    max_concurrency: int = 5,
    min_text_length: int = 50,
    use_ocr_if_needed: bool = True,
    skip_openai: bool = False,
    auto_ocr_detect: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)
    md = MarkItDown()

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pages_processed = 0

    if auto_ocr_detect:
        if progress_callback:
            progress_callback("Detecting if OCR is needed...", 0, 0)

        ocr_needed = check_if_ocr_needed(input_path, min_text_length=min_text_length)
        if ocr_needed:
            use_ocr_if_needed = True
            if progress_callback:
                progress_callback("OCR required - enabling OCR for all pages", 0, 0)
        else:
            use_ocr_if_needed = False
            if progress_callback:
                progress_callback("Text extraction sufficient - OCR not needed", 0, 0)

    if is_image_file(input_path):
        page_no = 1
        png_path = out_dir / f"page-{page_no:03d}.png"
        md_path = out_dir / f"page-{page_no:03d}.md"

        if png_path.exists() and md_path.exists():
            page_jobs = []
        else:
            from shutil import copy2
            copy2(str(input_path), str(png_path))

            if use_ocr_if_needed:
                raw_md = ocr_image(input_path)
            else:
                raw_md = ""

            page_jobs = [(page_no, md_path, png_path, raw_md, [])]
        total_pages = 1

    else:
        pdf_doc = pdfium.PdfDocument(str(input_path))
        total_pages = len(pdf_doc)

        page_jobs: List[Tuple[int, Path, Path, str, List[str]]] = []
        for i in range(total_pages):
            page_no = i + 1
            png_path = out_dir / f"page-{page_no:03d}.png"
            md_path = out_dir / f"page-{page_no:03d}.md"

            if png_path.exists() and md_path.exists():
                continue

            page_pdf_bytes = extract_page_pdf_bytes(input_path, i)
            raw_md = markitdown_convert_page(md, page_pdf_bytes)
            render_page_png(pdf_doc, i, png_path, dpi=dpi)

            extracted_images = extract_images_from_page(pdf_doc, i, out_dir, page_no)

            if use_ocr_if_needed and len(raw_md.strip()) < min_text_length:
                ocr_text = ocr_image(png_path)
                raw_md = ocr_text

            page_jobs.append((page_no, md_path, png_path, raw_md, extracted_images))

    total_jobs = len(page_jobs)

    if skip_openai:
        for page_no, md_path, png_path, raw_md, extracted_images in page_jobs:
            cleaned = raw_md
            with md_path.open("w", encoding="utf-8") as f:
                f.write(cleaned.strip() + "\n")
            pages_processed += 1

            if progress_callback:
                progress_callback(f"Processing page {page_no}", pages_processed, total_jobs)
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
            futures = {
                pool.submit(
                    ai_cleanup_page,
                    model,
                    page_no,
                    raw_md,
                    png_path,
                    temperature,
                    extracted_images,
                ): (page_no, md_path, png_path)
                for (page_no, md_path, png_path, raw_md, extracted_images) in page_jobs
            }

            for fut in as_completed(futures):
                page_no, md_path, png_path = futures[fut]
                try:
                    _page_no, cleaned, usage = fut.result()
                except Exception as e:
                    raw_md = next(r for (p, _, _, r, _) in page_jobs if p == page_no)
                    cleaned = raw_md
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                totals["completion_tokens"] += usage.get("completion_tokens", 0)
                totals["total_tokens"] += usage.get("total_tokens", 0)

                with md_path.open("w", encoding="utf-8") as f:
                    f.write(cleaned.strip() + "\n")

                pages_processed += 1

                if progress_callback:
                    progress_callback(f"Processed page {page_no}", pages_processed, total_jobs)

    combined = combine_pages(out_dir)

    return {
        "success": True,
        "output_dir": str(out_dir),
        "combined_file": str(combined),
        "token_usage": totals,
        "pages_processed": pages_processed,
        "total_pages": total_pages,
    }
