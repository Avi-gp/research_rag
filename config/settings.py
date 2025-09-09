import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Google API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")    
    # Paths
    VECTOR_DB_PATH: str ='./data/vector_db'
    PDF_STORAGE_PATH: str ='./data/pdfs'
    
    # Model settings
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 32768
    
    def __post_init__(self):
        Path(self.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.__post_init__()