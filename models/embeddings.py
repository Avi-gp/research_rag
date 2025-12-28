from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from config.settings import settings
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    NVIDIA Embeddings Model wrapper for LangChain RAG.
    """
    
    def __init__(self):
        """
        Initialize the NVIDIA embedding model.
        
        The truncate parameter controls how the model handles inputs that exceed
        the maximum token length:
        - "NONE": (default) Raises an error if input is too long
        - "START": Truncates from the beginning
        - "END": Truncates from the end
        
        For RAG applications where documents are pre-chunked, "NONE" is recommended
        to catch potential chunking issues early.
        """
        try:
            # Initialize NVIDIA embeddings
            # Note: API key can be set via NVIDIA_API_KEY environment variable
            # or passed directly as api_key parameter
            self.embedding_model = NVIDIAEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.NVIDIA_API_KEY,
                base_url=settings.NVIDIA_BASE_URL,
                truncate="NONE",  # Raise error if input exceeds max length
            )
        
            # Test the model with a sample text to ensure it's working
            # and get the embedding dimension
            test_embedding = self.embedding_model.embed_query(
                "Test connection and check embedding dimension"
            )
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"✅ Embedding model initialized: {settings.EMBEDDING_MODEL}")
            logger.info(f"✅ Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing embedding model: {str(e)}")
            raise

    def get_embeddings(self):
        """
        Return the embedding model instance for use with vector stores.
        
        Returns:
            NVIDIAEmbeddings: The embedding model instance
        """
        return self.embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text (query).
        
        This method is used for embedding queries in RAG applications.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If input text is empty
            Exception: If embedding generation fails
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
        Generate embeddings for multiple documents (passages).
        
        This method is used for embedding document chunks in RAG applications.
        
        Args:
            documents (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If document list is empty or contains only empty documents
            Exception: If embedding generation fails
        """
        try:
            if not documents:
                raise ValueError("Document list cannot be empty")
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc and doc.strip()]
            
            if not valid_documents:
                raise ValueError("No valid documents found (all are empty or whitespace)")
            
            if len(valid_documents) != len(documents):
                logger.warning(
                    f"Filtered out {len(documents) - len(valid_documents)} empty documents"
                )
            
            logger.info(f"Generating embeddings for {len(valid_documents)} documents...")
            embeddings = self.embedding_model.embed_documents(valid_documents)
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings for documents: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        This is useful for configuring vector stores and understanding
        the size of the embedding space.
        
        Returns:
            int: Embedding dimension
        """
        return self.embedding_dimension
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronous version of embed_text for query embeddings.
        
        Useful for applications that need async/await patterns.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            
            logger.info("Generating async embedding for query...")
            embedding = await self.embedding_model.aembed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating async embedding for query: {str(e)}")
            raise
    
    async def aembed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Asynchronous version of embed_documents for document embeddings.
        
        Useful for applications that need async/await patterns.
        
        Args:
            documents (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            if not documents:
                raise ValueError("Document list cannot be empty")
            
            valid_documents = [doc for doc in documents if doc and doc.strip()]
            
            if not valid_documents:
                raise ValueError("No valid documents found")
            
            if len(valid_documents) != len(documents):
                logger.warning(
                    f"Filtered out {len(documents) - len(valid_documents)} empty documents"
                )
            
            logger.info(f"Generating async embeddings for {len(valid_documents)} documents...")
            embeddings = await self.embedding_model.aembed_documents(valid_documents)
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating async embeddings for documents: {str(e)}")
            raise