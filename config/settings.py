import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Google API (for LLM only)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")  
    
    # NVIDIA API (for embeddings)
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")
    
    # Paths
    VECTOR_DB_PATH: str = './data/vector_db'
    PDF_STORAGE_PATH: str = './data/pdfs'
    LOGS_PATH: str = './logs'
    
    # Model settings
    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5" 
    LLM_MODEL: str = "gemini-2.5-flash" 
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 32768
    
    # NVIDIA specific settings
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    
    def __post_init__(self):
        # Validate required API keys
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LLM")
        if not self.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is required for embeddings")
            
        # Create directories
        Path(self.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.__post_init__()