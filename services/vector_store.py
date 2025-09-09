from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from models.embeddings import EmbeddingModel
from config.settings import settings

class VectorStore:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.embeddings = self.embedding_model.get_embeddings()
        self.vector_db_path = Path(settings.VECTOR_DB_PATH)
        self.vector_store: Optional[FAISS] = None
        
        # Load existing store if available
        self.load_store()
    
    def create_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store"""
        if not documents:
            print("No documents to index")
            return
        
        print(f"Creating vector index with {len(documents)} documents...")
        
        try:
            print("Generating embeddings for documents...")
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.save_store()
            print(f"✅ Vector index created with {len(documents)} documents")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing store"""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents...")
        
        try:
            if self.vector_store is None:
                self.create_store(documents)
            else:
                self.vector_store.add_documents(documents)
                self.save_store()
                print(f"✅ Added {len(documents)} documents")
                
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    def as_retriever(self, k: int = 5, threshold: float = 0.5) -> VectorStoreRetriever:
        """Get LangChain retriever with similarity search and threshold"""
        if self.vector_store is None:
            raise ValueError("No vector store available")
        
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": threshold}
        )
    
    def get_relevant_documents(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get relevant documents using similarity search with threshold"""
        if self.vector_store is None:
            print("No vector store available")
            return []
        
        try:
            # Use similarity_search_with_score directly for better control
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by threshold and convert to our standard format
            results = []
            for i, (doc, distance) in enumerate(docs_with_scores):
                # Convert distance to similarity score (1 - normalized_distance)
                # For cosine distance, similarity = 1 - distance
                similarity_score = max(0.0, 1.0 - distance)
                
                # Apply threshold filter
                if similarity_score >= threshold:
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "rank": i + 1,
                        "similarity_score": float(similarity_score)
                    }
                    results.append(result)
            
            print(f"Found {len(results)} documents above threshold {threshold}")
            return results
            
        except Exception as e:
            print(f"❌ Retrieval error: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Similarity search with scores and threshold filtering"""
        if self.vector_store is None:
            print("No vector store available")
            return []
        
        try:
            # Use the primary method for consistency
            return self.get_relevant_documents(query, k, threshold)
            
        except Exception as e:
            print(f"❌ Similarity search error: {str(e)}")
            return []
    
    def save_store(self) -> None:
        """Save vector store"""
        if self.vector_store is None:
            return
        
        try:
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.vector_db_path))
            print("✅ Vector store saved")
            
        except Exception as e:
            print(f"❌ Save error: {str(e)}")
            raise
    
    def load_store(self) -> None:
        """Load existing vector store"""
        try:
            index_path = self.vector_db_path / "index.faiss"
            if index_path.exists():
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    str(self.vector_db_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded")
            else:
                print("No existing vector store found")
                
        except Exception as e:
            print(f"❌ Load error: {str(e)}")
            self.vector_store = None
    
    def clear_store(self) -> Dict[str, Any]:
        """Clear vector store"""
        try:
            doc_count = self.get_document_count()
            
            self.vector_store = None
            
            if self.vector_db_path.exists():
                import shutil
                shutil.rmtree(self.vector_db_path)
            
            return {
                "status": "success",
                "documents_cleared": doc_count,
                "message": "Vector store cleared"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "documents_cleared": 0
            }
    
    def get_document_count(self) -> int:
        """Get total document count"""
        if self.vector_store is None:
            return 0
        
        try:
            return self.vector_store.index.ntotal
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.vector_store is None:
            return {
                "total_documents": 0,
                "store_exists": False,
                "vector_db_path": str(self.vector_db_path)
            }
        
        try:
            return {
                "total_documents": self.get_document_count(),
                "store_exists": True,
                "embedding_dimension": self.vector_store.index.d,
                "search_method": "similarity_score_threshold",
                "vector_db_path": str(self.vector_db_path),
                "index_exists": (self.vector_db_path / "index.faiss").exists()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def search_by_source(self, source: str) -> List[Document]:
        """Get documents by source"""
        if self.vector_store is None:
            return []
        
        try:
            # Simple approach: search all and filter
            all_docs = []
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore.search(doc_id)
                if doc and doc.metadata.get('source') == source:
                    all_docs.append(doc)
            return all_docs
            
        except Exception as e:
            print(f"❌ Source search error: {str(e)}")
            return []