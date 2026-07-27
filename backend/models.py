from pydantic import BaseModel, Field
from typing import Optional

class Base64_OCR_Request(BaseModel):
    image_base64: str=Field(..., description="Base64 image string (with or without URI header)")
    mime_type: Optional[str] = Field("image/jpeg", description="Image MIME type (image/png, image/jpeg, etc.)")
    prompt: Optional[str] = Field(None, description="Custom prompt instructions for the extraction")
    
class OCR_Response(BaseModel):
    success: bool
    text: str
    model_used: str
    error: Optional[str] = None
    
