import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from config import MODEL_NAME
from models import OCR_Response, Base64_OCR_Request
from ocr import process_image_ocr

app = FastAPI(
    title = "Gemma OCR Engine",
    description="Asynchronous FastAPI OCR service powered by Gemma multimodal on Openrouter.",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "online", "model": MODEL_NAME}

@app.post("api/v1/ocr/upload", response_model= OCR_Response)
async def ocr_from_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """Endpoint for drag and drop file upload (multipart/form-data)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = "Uploaded file must be an image (PNG, JPEG etc.)"
        )
    try:
        contents = await file.read()
        extracted_text = await process_image_ocr(
            image_bytes= contents,
            mime_type= file.content_type,
            custom_prompt= prompt
        )
        return OCR_Response(success= True, text= extracted_text, model_used= MODEL_NAME)
    
    except Exception as e:
        raise HTTPException(
            status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e)
        )

@app.post("/api/v1/ocr/base64", response_model= OCR_Response)
async def ocr_from_base64(payload: Base64_OCR_Request):
    # Endpoint for raw Base64 strings
    try:
        raw_base64 = payload.image_base64
        mime_type = payload.mime_type or "image/jpeg"
        
        # Auto-clean data URI headers if the frontend sends "data:image/png;base64,..."
        if "," in raw_base64:
            header, raw_base64 = raw_base64.split(",", 1)
            if "data" in header and ";base64" in header:
                mime_type = header.split(";")[0].replace("data", "")
                
        image_bytes = base64.b64decode(raw_base64)
        
        extracted_text = await process_image_ocr(
            image_bytes=image_bytes,
            mime_type=mime_type,
            custom_prompt=payload.prompt
        )
        return OCR_Response(success=True, text=extracted_text, model_used=MODEL_NAME)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    