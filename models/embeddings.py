from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings
import google.generativeai as genai
from typing import List, Union

class EmbeddingModel:
    def __init__(self):
        """Initialize the Google Generative AI embedding model"""
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY
            )
            print(f"✓ Embedding model initialized: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"✗ Error initializing embedding model: {str(e)}")
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
            
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            print(f"✗ Error generating embedding for text: {str(e)}")
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
                print(f"Warning: Filtered out {len(documents) - len(valid_documents)} empty documents")
            
            embeddings = self.embedding_model.embed_documents(valid_documents)
            return embeddings
        except Exception as e:
            print(f"✗ Error generating embeddings for documents: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            int: Embedding dimension
        """
        try:
            # Test with a sample text to get dimension
            sample_embedding = self.embed_text("sample text for dimension check")
            return len(sample_embedding)
        except Exception as e:
            print(f"✗ Error getting embedding dimension: {str(e)}")
            return 0