import base64
import httpx
from config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_URL

DEFAULT_PROMPT = (
    "Extract all legible text from this image."
    "Preserve all structural formatting like headers, bullet points, and tables in Markdown."
    "Do not add introductory conversational filler-return only extracted text."
)

async def process_image_ocr(
    image_bytes: bytes,
    mime_type: str,
    custom_prompt: str = None
) -> str: 
    """Encodes raw image bytes into base64 and sends an HTTP request to OpenRouter."""
    
    # Converts raw image data to Base64 data URI
    encoded_b64 = base64.b64decode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type}; base64, {encoded_b64}"
    
    #build payload as per openrouter instructions
    prompt_text = custom_prompt if custom_prompt else DEFAULT_PROMPT
    
    payload = {
        "model": MODEL_NAME,
        "messages":[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    "type": "image_url", "image_url":{"url": data_uri}
                ]
            }
        ],
        "temperature": 0.1 #low temp for deterministic AI (no guesswork)
    }