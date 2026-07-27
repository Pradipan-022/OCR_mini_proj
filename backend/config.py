import os  
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

