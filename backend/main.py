import io
import base64
import asyncio
import traceback
from typing import Optional

import numpy as np
import pytesseract
import easyocr
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Import existing backend modules
from backend.config import MODEL_NAME
from backend.models import Base64_OCR_Request, OCR_Response
from backend.ocr import process_image_ocr


# --- 1. Initialize Global EasyOCR Reader (Loaded once into RAM) ---
# Set gpu=False for standard CPU execution on Linux Mint
easyocr_reader = easyocr.Reader(['en'], gpu=False)


# --- 2. Synchronous Helper Functions for Local Engines ---
def run_tesseract_ocr(image_bytes: bytes) -> str:
    """CPU-bound task for local Tesseract execution."""
    image = Image.open(io.BytesIO(image_bytes))
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()


def run_easyocr_ocr(image_bytes: bytes) -> str:
    """CPU-bound PyTorch task for local EasyOCR execution."""
    # Convert image bytes to a RGB PIL image, then to a NumPy array for OpenCV/PyTorch
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # detail=0 returns plain text strings; paragraph=True groups nearby lines together
    results = easyocr_reader.readtext(image_np, detail=0, paragraph=True)
    return "\n".join(results).strip()


# --- 3. FastAPI Setup ---
app = FastAPI(
    title="Multi-Engine OCR API",
    description="Asynchronous FastAPI OCR supporting Gemma (Cloud), Tesseract (Local C++), and EasyOCR (Local PyTorch).",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. API Endpoints ---
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "available_engines": ["gemma", "tesseract", "easyocr"],
        "gemma_model": MODEL_NAME
    }


@app.post("/api/v1/ocr/upload", response_model=OCR_Response)
async def ocr_from_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    engine: str = Form("gemma")
):
    """Multipart form-data endpoint handling file uploads."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image (PNG, JPEG, etc.)"
        )

    try:
        contents = await file.read()
        selected_engine = engine.lower()

        # Engine Routing Logic
        if selected_engine == "easyocr":
            extracted_text = await asyncio.to_thread(run_easyocr_ocr, contents)
            used_model = "EasyOCR (Local PyTorch)"

        elif selected_engine == "tesseract":
            extracted_text = await asyncio.to_thread(run_tesseract_ocr, contents)
            used_model = "Tesseract OCR (Local C++)"

        else:
            extracted_text = await process_image_ocr(
                image_bytes=contents,
                mime_type=file.content_type,
                custom_prompt=prompt
            )
            used_model = f"Gemma ({MODEL_NAME})"

        return OCR_Response(
            success=True,
            text=extracted_text,
            model_used=used_model
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/ocr/base64", response_model=OCR_Response)
async def ocr_from_base64(payload: Base64_OCR_Request):
    """JSON endpoint handling raw Base64 strings."""
    try:
        raw_base64 = payload.image_base64
        mime_type = payload.mime_type or "image/jpeg"

        if "," in raw_base64:
            header, raw_base64 = raw_base64.split(",", 1)
            if "data" in header and ";base64" in header:
                mime_type = header.split(";")[0].replace("data", "")

        image_bytes = base64.b64decode(raw_base64)
        selected_engine = (payload.engine or "gemma").lower()

        if selected_engine == "easyocr":
            extracted_text = await asyncio.to_thread(run_easyocr_ocr, image_bytes)
            used_model = "EasyOCR (Local PyTorch)"

        elif selected_engine == "tesseract":
            extracted_text = await asyncio.to_thread(run_tesseract_ocr, image_bytes)
            used_model = "Tesseract OCR (Local C++)"

        else:
            extracted_text = await process_image_ocr(
                image_bytes=image_bytes,
                mime_type=mime_type,
                custom_prompt=payload.prompt
            )
            used_model = f"Gemma ({MODEL_NAME})"

        return OCR_Response(
            success=True,
            text=extracted_text,
            model_used=used_model
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)