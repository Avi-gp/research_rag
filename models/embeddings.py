from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from config.settings import settings
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        """Initialize the NVIDIA embedding model"""
        try:
            # Initialize NVIDIA embeddings without truncate parameter
            # Since documents will be pre-chunked for RAG
            self.embedding_model = NVIDIAEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.NVIDIA_API_KEY,
                base_url=settings.NVIDIA_BASE_URL,
                truncate="NONE",
            )
        
            # Test the model with a sample text to ensure it's working
            test_embedding = self.embedding_model.embed_query("Test connection and Check Embedding Dimension")
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"✅ Embedding model initialized: {settings.EMBEDDING_MODEL}")
            logger.info(f"✅ Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing embedding model: {str(e)}")
            raise

    def get_embeddings(self):
        """Return the embedding model instance"""
        return self.embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            logger.info("Generating embedding for input text...")
            embedding = self.embedding_model.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding for text: {str(e)}")
            raise
            
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            documents (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            if not documents:
                raise ValueError("Document list cannot be empty")
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc and doc.strip()]
            
            if not valid_documents:
                raise ValueError("No valid documents found (all are empty or whitespace)")
            
            if len(valid_documents) != len(documents):
                logger.warning(f"Filtered out {len(documents) - len(valid_documents)} empty documents")
            
            logger.info(f"Generating embeddings for {len(valid_documents)} documents...")
            embeddings = self.embedding_model.embed_documents(valid_documents)
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings for documents: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            int: Embedding dimension
        """
        return self.embedding_dimension