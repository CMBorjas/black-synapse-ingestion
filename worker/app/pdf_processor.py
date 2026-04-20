"""
PDF Processor for AtlasAI

Extracts text and images from PDF files.
Images are described using a vision model (moondream via Ollama by default,
with optional fallback to OpenAI GPT-4o Vision).

The output is a single combined text string structured as:

  [Page 1 Text]
  ...page text...

  [Page 1 Image 1 Description]
  ...vision model description...

  ---

  [Page 2 Text]
  ...

This combined text is then fed into the normal ingestion pipeline
(chunk → embed → upsert to Qdrant).
"""

import base64
import io
import logging
import os
from typing import Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Ollama config (reads from env, falls back to defaults)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")

# Minimum image size to bother describing (skip tiny icons/bullets)
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50


async def describe_image_with_ollama(img: Image.Image, model: str = VISION_MODEL) -> str:
    """
    Send an image to an Ollama vision model and return a text description.

    Args:
        img:   PIL Image object to describe
        model: Ollama model name (must support vision, e.g. moondream, llava)

    Returns:
        Description string, or empty string if the call fails.
    """
    # Convert image to PNG bytes then base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "Describe this image in detail. "
        "Include any visible text, labels, numbers, chart data, diagram components, "
        "or other information that would help someone understand the document without seeing the image."
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [img_b64],
                        }
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Ollama vision (%s) failed for image: %s", model, exc)
        return ""


async def describe_image_with_openai(img: Image.Image, openai_client) -> str:
    """
    Fallback: describe an image using OpenAI GPT-4o Vision.

    Args:
        img:           PIL Image object
        openai_client: Configured openai.OpenAI client instance

    Returns:
        Description string, or empty string on failure.
    """
    import asyncio

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{img_b64}"

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail, including any visible text, "
                                "charts, diagrams, or data that is relevant to understanding the document."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("OpenAI vision fallback failed: %s", exc)
        return ""


async def extract_pdf_content(
    pdf_bytes: bytes,
    openai_client=None,
    use_openai_vision: bool = False,
) -> str:
    """
    Extract all content from a PDF, combining text and image descriptions.

    For each page:
      1. Extract raw text via PyMuPDF
      2. Extract embedded images
      3. Describe each image using a vision model
      4. Combine text + descriptions into a structured string

    Args:
        pdf_bytes:          Raw PDF file bytes
        openai_client:      Optional OpenAI client for GPT-4o Vision fallback
        use_openai_vision:  If True, use GPT-4o Vision instead of Ollama moondream

    Returns:
        Combined content string ready for the ingestion pipeline.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Add 'pymupdf>=1.23.0' to requirements.txt "
            "and reinstall dependencies."
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_sections = []

    logger.info("Processing PDF: %d pages", len(doc))

    for page_num, page in enumerate(doc, start=1):
        page_parts = []

        # ── Text extraction ────────────────────────────────────────────────────
        text = page.get_text("text").strip()
        if text:
            page_parts.append(f"[Page {page_num} Text]\n{text}")

        # ── Image extraction ───────────────────────────────────────────────────
        image_list = page.get_images(full=True)
        logger.debug("Page %d: found %d image(s)", page_num, len(image_list))

        for img_index, img_info in enumerate(image_list, start=1):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                # Open with Pillow so we can normalise format and filter tiny images
                img = Image.open(io.BytesIO(img_bytes))

                # Skip very small images (icons, decorative elements, etc.)
                if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                    logger.debug(
                        "Skipping tiny image %d on page %d (%dx%d)",
                        img_index, page_num, img.width, img.height,
                    )
                    continue

                # Normalise colour mode (PDFs can have CMYK, grayscale, etc.)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                logger.info(
                    "Describing image %d on page %d (%dx%d) via %s",
                    img_index, page_num, img.width, img.height,
                    "OpenAI" if use_openai_vision else "Ollama",
                )

                if use_openai_vision and openai_client:
                    description = await describe_image_with_openai(img, openai_client)
                else:
                    description = await describe_image_with_ollama(img)

                if description:
                    page_parts.append(
                        f"[Page {page_num} Image {img_index} Description]\n{description}"
                    )
                else:
                    logger.warning(
                        "No description returned for image %d on page %d", img_index, page_num
                    )

            except Exception as exc:
                logger.warning(
                    "Failed to process image %d on page %d: %s", img_index, page_num, exc
                )
                continue

        if page_parts:
            page_sections.append("\n\n".join(page_parts))

    doc.close()

    if not page_sections:
        logger.warning("No extractable content found in PDF")
        return ""

    combined = "\n\n---\n\n".join(page_sections)
    logger.info(
        "PDF extraction complete: %d page section(s), %d chars total",
        len(page_sections), len(combined),
    )
    return combined
