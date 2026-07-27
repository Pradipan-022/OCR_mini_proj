import os  
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31b-it:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

