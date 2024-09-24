import os
from typing import List, Optional

from dotenv import find_dotenv

class Settings():
    # Path Settings
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    PDF_CACHE_DIR = os.path.join(BASE_DIR, "cache")  # Where to save pdfs
    DATA_DIR = os.path.join(BASE_DIR, "data")  # Where to save data
    DEBUG: bool = True
    
    LLM_TEMPERATURE: float = 0.5
    LLM_TIMEOUT: int = 480
    LLM_MAX_RESPONSE_TOKENS: int = 2048
    
    OPENAI_BASE_URL: Optional[str] = 'http://localhost:11434/v1'
    OPENAI_API_KEY: Optional[str] = 'ollama'
    LLM_TYPE: str = "llama3.1"
    
    class Config:
        env_file = find_dotenv(".env")
    
settings = Settings()