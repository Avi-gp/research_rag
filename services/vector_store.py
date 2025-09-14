from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain.schema import Document
from langchain_chroma import Chroma
from models.embeddings import EmbeddingModel
from config.settings import settings
import chromadb
from chromadb.config import Settings as ChromaSettings
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    ChromaDB Vector Store implementation for document storage and semantic search.
    """
    
    def __init__(self):
        """Initialize the ChromaDB vector store"""
        self.embedding_model = EmbeddingModel()
        self.embeddings = self.embedding_model.get_embeddings()
        self.embedding_dimension = self.embedding_model.get_embedding_dimension()
        self.vector_db_path = Path(settings.VECTOR_DB_PATH)
        self.collection_name = "research_paper_rag"
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.vector_store: Optional[Chroma] = None
        
        # Initialize ChromaDB client and load existing collection
        self._initialize_chroma_client()
        self._load_existing_collection()
    
    def _initialize_chroma_client(self) -> None:
        """Initialize ChromaDB persistent client"""
        try:
            # Ensure the directory exists
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB client initialized at: {self.vector_db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _load_existing_collection(self) -> None:
        """Load existing ChromaDB collection if it exists"""
        try:
            if self.chroma_client is None:
                logger.warning("ChromaDB client not initialized")
                return
            
            # Check if collection exists
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if self.collection_name in existing_collections:
                logger.info("Loading existing vector collection...")
                
                # Create Chroma vector store instance using the existing client
                self.vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                
                doc_count = self.get_document_count()
                logger.info(f"Vector collection loaded with {doc_count} documents")
            else:
                logger.info("No existing vector collection found")
                
        except Exception as e:
            logger.error(f"Error loading vector collection: {str(e)}")
            self.vector_store = None
    
    def _ensure_collection_loaded(self) -> bool:
        """Ensure vector collection is loaded and available"""
        if self.vector_store is not None:
            return True
        
        # Try to load existing collection
        self._load_existing_collection()
        return self.vector_store is not None
    
    def create_collection(self, documents: List[Document]) -> bool:
        """
        Create a new ChromaDB collection
        
        Args:
            documents (List[Document]): List of documents to index
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided to create collection")
            return False
        
        try:
            logger.info(f"Creating ChromaDB collection with {len(documents)} documents...")
            
            if self.chroma_client is None:
                raise ValueError("ChromaDB client not initialized")
            
            # Create new Chroma vector store using the existing client
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                collection_metadata={
                    "title": "Research Paper RAG Collection",
                    "description": "Vector store for research paper retrieval-augmented generation",
                    "distance_metric": "cosine",
                },
                collection_configuration={"hnsw:space": "cosine"}
            )
            
            # Add documents to the collection
            self.vector_store.add_documents(documents)
            
            logger.info(f"Vector collection created successfully with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector collection: {str(e)}")
            return False
    
    def insert_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to existing vector collection
        
        Args:
            documents (List[Document]): Documents to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            if self.vector_store is None:
                logger.info("No existing collection found, creating new vector collection...")
                return self.create_collection(documents)
            else:
                logger.info(f"Adding {len(documents)} documents to existing vector collection...")
                self.vector_store.add_documents(documents)
                logger.info(f"Successfully added {len(documents)} documents")
                return True
                
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def docs_retrieval(
        self, 
        query: str, 
        k: int = 10,
        threshold: Optional[float] = None  # Made optional - None means no filtering
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using cosine similarity.
        
        Args:
            query (str): Search query
            k (int): Number of top results to retrieve
            threshold (Optional[float]): Minimum similarity score (0.0 to 1.0)
                                       If None, no threshold filtering is applied
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata
        """
        if not self._ensure_collection_loaded():
            logger.warning("No vector collection available for search")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            logger.info(f"Searching: '{query}' with k={k}, threshold={threshold}")
            
            # Use similarity_search_with_score for cosine similarity
            docs_with_distances = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            
            # Log raw results (documents with distances) for debugging/monitoring
            try:
                logger.info(f"Raw similarity_search_with_score returned {len(docs_with_distances)} items")
                for idx, (doc, distance) in enumerate(docs_with_distances, start=1):
                    cosine_sim = 1 - distance
                    sim_score = (cosine_sim + 1) / 2
                    sim_score = max(0.0, min(1.0, sim_score))
                    # Truncate content for logging to avoid huge logs
                    snippet = getattr(doc, "page_content", "")[:200].replace("\n", " ")
                    metadata = getattr(doc, "metadata", {})
                    logger.info(
                        "Result %d: sim_score=%.4f, distance=%.4f, cosine_sim=%.4f, source=%s, chunk_id=%s, metadata=%s, snippet=%s",
                        idx,
                        sim_score,
                        distance,
                        cosine_sim,
                        metadata.get("source", "Unknown"),
                        metadata.get("chunk_id", f"chunk_{idx}"),
                        metadata,
                        snippet
                    )
            except Exception as log_exc:
                logger.warning(f"Failed to log search results: {log_exc}")
            
            # Convert distances to similarity scores
            results = []
            for rank, (doc, distance) in enumerate(docs_with_distances):
                # Convert distance to cosine similarity and then to normalized similarity
                cosine_similarity = 1 - distance
                similarity_score = (cosine_similarity + 1) / 2
                similarity_score = max(0.0, min(1.0, similarity_score))
                
                # Apply threshold filtering only if threshold is specified
                if threshold is not None and similarity_score < threshold:
                    continue
                
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rank": rank + 1,
                    "sim_score": round(similarity_score, 4),
                    "distance": round(distance, 4),
                    "cosine_sim": round(cosine_similarity, 4),
                    "source": doc.metadata.get('source', 'Unknown'),
                    "chunk_id": doc.metadata.get('chunk_id', f'chunk_{rank + 1}')
                }
                results.append(result)
            
            if threshold is not None:
                logger.info(f"Found {len(results)} documents above threshold {threshold}")
            else:
                logger.info(f"Found {len(results)} documents (no threshold filtering)")
                
            if results:
                avg_score = sum(r['sim_score'] for r in results) / len(results)
                logger.info(f"Average similarity score: {avg_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear the vector collection and remove all data
        
        Returns:
            Dict[str, Any]: Status of the clear operation
        """
        try:
            doc_count = self.get_document_count()
            
            if self.chroma_client is None:
                raise ValueError("ChromaDB client not initialized")
            
            # Delete the collection if it exists
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if self.collection_name in existing_collections:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' deleted")
            except Exception as e:
                logger.warning(f"Error deleting collection: {str(e)}")
            
            # Clear in-memory store reference
            self.vector_store = None
            
            logger.info(f"Vector collection cleared ({doc_count} documents removed)")
            
            return {
                "status": "success",
                "documents_cleared": doc_count,
                "message": f"Vector collection cleared successfully. {doc_count} documents removed."
            }
            
        except Exception as e:
            logger.error(f"Error clearing vector collection: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "documents_cleared": 0
            }
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in the vector collection
        
        Returns:
            int: Number of documents
        """
        if self.chroma_client is None:
            return 0
        
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            if self.collection_name not in existing_collections:
                return 0
            
            collection = self.chroma_client.get_collection(name=self.collection_name)
            return collection.count()
            
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector collection
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        try:
            stats = {
                "collection_exists": self.vector_store is not None,
                "vector_db_path": str(self.vector_db_path),
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimension": self.embedding_model.get_embedding_dimension(),
                "collection_name": self.collection_name,
                "vector_store_type": "ChromaDB",
                "distance_strategy": "COSINE",
                "similarity_type": "Semantic Search"
            }
            
            if self.chroma_client is not None:
                try:
                    existing_collections = [col.name for col in self.chroma_client.list_collections()]
                    if self.collection_name in existing_collections:
                        collection = self.chroma_client.get_collection(name=self.collection_name)
                        stats.update({
                            "total_documents": self.get_document_count(),
                            "collection_available": True,
                            "search_method": "similarity_search_with_score",
                            "client_type": "PersistentClient",
                            "storage_path": str(self.vector_db_path),
                            "collection_metadata": collection.metadata if hasattr(collection, 'metadata') else {},
                            "threshold_recommendations": {
                                "high_quality": 0.7,      # Lowered from 0.8
                                "good_quality": 0.5,      # Lowered from 0.7
                                "moderate_quality": 0.3,  # Lowered from 0.6
                                "low_quality": 0.2        # Lowered from 0.5
                            }
                        })
                    else:
                        stats.update({
                            "total_documents": 0,
                            "collection_available": False,
                            "search_method": None
                        })
                except Exception as e:
                    logger.warning(f"Error getting detailed collection stats: {str(e)}")
                    stats.update({
                        "total_documents": 0,
                        "collection_available": False,
                        "error": str(e)
                    })
            else:
                stats.update({
                    "total_documents": 0,
                    "collection_available": False,
                    "search_method": None,
                    "client_type": None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def search_by_source(self, source: str, k: int = 100) -> List[Document]:
        """
        Get documents by source file from the collection
        
        Args:
            source (str): Source file path/name to search for
            k (int): Maximum number of documents to return
            
        Returns:
            List[Document]: Documents from the specified source
        """
        # Ensure collection is loaded
        if not self._ensure_collection_loaded():
            logger.warning("No vector collection available for source search")
            return []
        
        try:
            # Try ChromaDB's native metadata filtering first
            try:
                results = self.vector_store.similarity_search(
                    query="",  # Empty query since we're filtering by metadata
                    k=k,
                    filter={"source": source}  # ChromaDB metadata filtering
                )
                logger.info(f"Found {len(results)} documents from source: {source}")
                return results
            except Exception as filter_error:
                logger.warning(f"Native metadata filtering failed: {str(filter_error)}")
                # Fallback to manual filtering
                pass
            
            # Fallback: Get all documents and filter manually
            # Use a generic query to get documents
            all_docs = self.vector_store.similarity_search(
                query="content",  # Generic query
                k=k * 3  # Get more to account for filtering
            )
            
            matching_docs = [
                doc for doc in all_docs 
                if doc.metadata.get('source') == source
            ][:k]  # Limit to k results
            
            logger.info(f"Found {len(matching_docs)} documents from source (fallback): {source}")
            return matching_docs
            
        except Exception as e:
            logger.error(f"Error in source search: {str(e)}")
            return []
    
    def is_collection_available(self) -> bool:
        """
        Check if vector collection is available and ready for use
        
        Returns:
            bool: True if collection is available, False otherwise
        """
        return (self._ensure_collection_loaded() and 
                self.get_document_count() > 0)
    
    def get_all_sources(self) -> List[str]:
        """
        Get list of all unique sources in the vector collection
        
        Returns:
            List[str]: List of unique source names
        """
        if not self._ensure_collection_loaded():
            return []
        
        try:
            # Get a large sample of documents to find sources
            docs = self.vector_store.similarity_search(
                query="content document", 
                k=1000  # Large k to get comprehensive source list
            )
            
            sources = set()
            for doc in docs:
                source = doc.metadata.get('source')
                if source:
                    sources.add(source)
            
            return list(sources)
            
        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            return []
    
    def delete_documents_by_source(self, source: str) -> Dict[str, Any]:
        """
        Delete all documents from a specific source in the collection
        
        Args:
            source (str): Source to delete documents from
            
        Returns:
            Dict[str, Any]: Status of deletion operation
        """
        if not self._ensure_collection_loaded():
            return {
                "status": "error",
                "message": "No vector collection available",
                "documents_deleted": 0
            }
        
        try:
            # First, find documents from this source
            docs_to_delete = self.search_by_source(source)
            
            if not docs_to_delete:
                return {
                    "status": "success",
                    "message": f"No documents found from source: {source}",
                    "documents_deleted": 0
                }
            
            # Extract document IDs if available
            doc_ids = []
            for doc in docs_to_delete:
                doc_id = doc.metadata.get('id') or doc.metadata.get('doc_id')
                if doc_id:
                    doc_ids.append(doc_id)
            
            if doc_ids:
                # Use ChromaDB client for deletion
                collection = self.chroma_client.get_collection(name=self.collection_name)
                collection.delete(ids=doc_ids)
                
                return {
                    "status": "success",
                    "message": f"Deleted {len(doc_ids)} documents from source: {source}",
                    "documents_deleted": len(doc_ids)
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Found documents from {source} but no IDs to delete. Consider recreating the vector collection.",
                    "documents_deleted": 0
                }
            
        except Exception as e:
            logger.error(f"Error deleting documents by source: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting documents: {str(e)}",
                "documents_deleted": 0
            }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'chroma_client') and self.chroma_client:
                # ChromaDB handles cleanup automatically for persistent clients
                pass
        except Exception:
            pass  # Ignore cleanup errors