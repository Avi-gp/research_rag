import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.llm_service import LLMService
from config.settings import settings
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class RAGEngine:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
    
    
    def ingest_documents(self, pdf_paths: List[str] = None, pdf_directory: str = None) -> Dict[str, Any]:
        """Ingest PDF documents into the RAG system"""
        try:
            # Use the new process_pdfs_from_directory method
            if pdf_directory:
                documents = self.pdf_processor.process_pdfs_from_directory(pdf_directory)
            elif pdf_paths:
                # For individual PDF paths, process the parent directories
                documents = []
                processed_dirs = set()
                for pdf_path in pdf_paths:
                    parent_dir = str(Path(pdf_path).parent)
                    if parent_dir not in processed_dirs:
                        # This will process all PDFs in the directory, but with hash checking
                        dir_documents = self.pdf_processor.process_pdfs_from_directory(parent_dir)
                        documents.extend(dir_documents)
                        processed_dirs.add(parent_dir)
            else:
                # Default to PDF storage path
                documents = self.pdf_processor.process_pdfs_from_directory(settings.PDF_STORAGE_PATH)
            
            if documents:
                self.vector_store.insert_documents(documents)
                
            return {
                "status": "success",
                "message": f"Successfully ingested {len(documents)} document chunks",
                "documents_processed": len(documents)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error ingesting documents: {str(e)}",
                "documents_processed": 0
            }
    
    def search_documents(self, query: str, k: int = 10, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using similarity search"""
        try:
            results = self.vector_store.docs_retrieval(query=query, k=k, threshold=threshold)
            return convert_numpy_types(results)
        except Exception as e:
            print(f"Error in search_documents: {str(e)}")
            return []
    
    def generate_answer(self, query: str, k: int = 10, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Generate answer using RAG pipeline"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_documents(query=query, k=k, threshold=threshold)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base. Please check if documents have been ingested.",
                    "sources": [],
                    "context_used": 0,
                    "search_method": "similarity_search",
                    "threshold_applied": threshold
                }
            
            # Combine context from relevant documents
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Generate answer
            answer = self.llm_service.answer_question(context, query)
            
            # Extract source information with type conversion
            sources = []
            for doc in relevant_docs:
                source_info = {
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "chunk_id": doc['metadata'].get('chunk_id', 'Unknown'),
                    "rank": int(doc.get('rank', 0)),
                    "similarity_score": float(doc.get('sim_score', 0.0)),
                    "distance": float(doc.get('distance', 1.0))
                }
                
                if source_info not in sources:
                    sources.append(source_info)
            
            return convert_numpy_types({
                "answer": answer,
                "sources": sources,
                "context_used": len(relevant_docs),
                "search_method": "similarity_search",
                "threshold_applied": threshold,
                "average_similarity": round(sum(doc.get('sim_score', 0.0) for doc in relevant_docs) / len(relevant_docs), 4) if relevant_docs else 0.0
            })
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "context_used": 0,
                "search_method": "similarity_search",
                "threshold_applied": threshold
            }
    
    def summarize_documents(self, query: str = None, k: int = 15, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Generate summary of documents using similarity search"""
        try:
            if query:
                # Search for relevant documents based on query
                relevant_docs = self.search_documents(query=query, k=k, threshold=threshold)
            else:
                # Get documents for general summary
                relevant_docs = self.search_documents(query="research summary overview", k=k, threshold=threshold)
            
            if not relevant_docs:
                threshold_msg = f" above threshold {threshold}" if threshold is not None else ""
                return {
                    "summary": f"No documents found{threshold_msg} for summarization in the knowledge base.",
                    "sources": [],
                    "context_used": 0,
                    "search_method": "similarity_search",
                    "threshold_applied": threshold
                }
            
            # Combine context
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Generate summary
            summary = self.llm_service.generate_summary(context, query)
            
            # Extract sources with counts and scores
            sources = {}
            for doc in relevant_docs:
                source = doc['metadata'].get('source', 'Unknown')
                sim_score = float(doc.get('sim_score', 0.0))
                
                if source in sources:
                    sources[source]['chunks'] += 1
                    sources[source]['total_score'] += sim_score
                    sources[source]['best_score'] = max(
                        sources[source]['best_score'], 
                        sim_score
                    )
                else:
                    sources[source] = {
                        'chunks': 1,
                        'total_score': sim_score,
                        'best_score': sim_score
                    }
            
            # Convert to list format with score metrics
            source_list = []
            for source, info in sources.items():
                source_info = {
                    "source": source,
                    "chunks_used": info['chunks'],
                    "avg_similarity": round(float(info['total_score'] / info['chunks']), 4) if info['chunks'] > 0 else 0.0,
                    "best_similarity": round(float(info['best_score']), 4)
                }
                source_list.append(source_info)
            
            return convert_numpy_types({
                "summary": summary,
                "sources": source_list,
                "context_used": len(relevant_docs),
                "search_method": "similarity_search",
                "threshold_applied": threshold,
                "average_similarity": round(sum(doc.get('sim_score', 0.0) for doc in relevant_docs) / len(relevant_docs), 4) if relevant_docs else 0.0
            })
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "sources": [],
                "context_used": 0,
                "search_method": "similarity_search",
                "threshold_applied": threshold
            }
    
    def search_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Search documents by source file name"""
        try:
            documents = self.vector_store.search_by_source(source_name)
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get('chunk_id', 'Unknown')
                }
                for doc in documents
            ]
            return convert_numpy_types(results)
        except Exception as e:
            print(f"Error searching by source: {str(e)}")
            return []
        
    def clear_pdf_storage(self) -> Dict[str, Any]:
        """Clear all PDF files from storage and reset processed files tracking"""
        try:
            pdf_storage_path = Path(settings.PDF_STORAGE_PATH)
            files_removed = []
            files_count = 0
            
            if pdf_storage_path.exists():
                # Remove all PDF files
                for file_path in pdf_storage_path.glob("*.pdf"):
                    file_path.unlink()
                    files_removed.append(file_path.name)
                    files_count += 1
                
                # Remove other files if any
                for file_path in pdf_storage_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        files_removed.append(file_path.name)
                        files_count += 1
            
            # Reset the PDF processor's processed files tracking
            self.pdf_processor.processed_files = {}
            self.pdf_processor._save_processed_files()
            
            return {
                "status": "success",
                "message": f"Cleared {files_count} files from PDF storage and reset tracking",
                "files_removed": files_removed,
                "files_count": files_count
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing PDF storage: {str(e)}",
                "files_removed": [],
                "files_count": 0
            }
    
    def reset_system(self, include_pdfs: bool = True) -> Dict[str, Any]:
        """Reset the entire RAG system"""
        try:
            results = {
                "status": "success",
                "message": "System reset successfully",
                "vector_store_reset": {},
                "pdf_storage_cleared": {},
                "processor_reset": True
            }
            
            # Reset vector store using the new method name
            vector_result = self.vector_store.clear_collection()
            results["vector_store_reset"] = vector_result
            
            if vector_result["status"] == "error":
                results["status"] = "partial_success"
                results["message"] = "System partially reset - vector store had issues"
            
            # Clear PDF storage if requested
            if include_pdfs:
                pdf_result = self.clear_pdf_storage()
                results["pdf_storage_cleared"] = pdf_result
                
                if pdf_result["status"] == "error":
                    results["status"] = "partial_success"
                    if results["message"] == "System reset successfully":
                        results["message"] = "System partially reset - PDF storage had issues"
                    else:
                        results["message"] = "System partially reset - both vector store and PDF storage had issues"
            
            # Reset PDF processor
            try:
                self.pdf_processor.processed_files = {}
                self.pdf_processor._save_processed_files()
            except Exception as e:
                print(f"Warning: Could not reset PDF processor: {str(e)}")
                results["processor_reset"] = False
            
            # Reinitialize components to ensure clean state
            try:
                self.vector_store = VectorStore()
                self.pdf_processor = PDFProcessor()
            except Exception as e:
                print(f"Warning: Could not reinitialize components: {str(e)}")
            
            return convert_numpy_types(results)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error resetting system: {str(e)}",
                "vector_store_reset": {},
                "pdf_storage_cleared": {},
                "processor_reset": False
            }
    
    def clear_vector_store_only(self) -> Dict[str, Any]:
        """Clear only the vector store, keeping PDF files"""
        try:
            result = self.vector_store.clear_collection()
            
            # Reset PDF processor processed files tracking so files will be reprocessed
            self.pdf_processor.processed_files = {}
            self.pdf_processor._save_processed_files()
            
            return convert_numpy_types({
                "status": result["status"],
                "message": f"{result['message']} and reset processing tracking",
                "documents_cleared": result.get("documents_cleared", 0)
            })
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing vector store: {str(e)}",
                "documents_cleared": 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get essential system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        
        # Get PDF storage stats
        pdf_storage_path = Path(settings.PDF_STORAGE_PATH)
        pdf_files_count = 0
        pdf_total_size = 0
        pdf_files_list = []
        
        if pdf_storage_path.exists():
            pdf_files = list(pdf_storage_path.glob("*.pdf"))
            pdf_files_count = len(pdf_files)
            pdf_total_size = sum(f.stat().st_size for f in pdf_files if f.exists())
            pdf_files_list = [f.name for f in pdf_files]
        
        # Get processed files info
        processed_files_info = self.pdf_processor.get_processed_files_info()
        
        stats = {
            "vector_store": {
                "status": "Ready" if vector_stats.get("collection_available", False) else "Not Ready",
                "total_documents": int(vector_stats.get("total_documents", 0)),
                "embedding_dimension": int(vector_stats.get("embedding_dimension", 0)),
                "collection_name": vector_stats.get("collection_name", "Unknown"),
                "vector_store_type": vector_stats.get("vector_store_type", "Unknown"),
                "distance_strategy": vector_stats.get("distance_strategy", "Unknown"),
                "storage_path": vector_stats.get("vector_db_path", "Unknown")
            },
            "pdf_processor": {
                "processed_files_count": len(processed_files_info),
                "processed_files_info": processed_files_info,
                "total_chunks_processed": sum(
                    info.get('chunks', 0) for info in processed_files_info.values()
                )
            },
            "pdf_storage": {
                "files_count": pdf_files_count,
                "total_size_mb": round(pdf_total_size / (1024 * 1024), 2),
                "storage_path": str(pdf_storage_path),
                "files_list": pdf_files_list
            },
            "search_capabilities": {
                "similarity_search": True,
                "source_based_search": True,
                "incremental_processing": True,
                "threshold_support": True,
                "cosine_similarity": True
            },
            "quality_thresholds": vector_stats.get("threshold_recommendations", {
                "high_quality": 0.7,
                "good_quality": 0.5,
                "moderate_quality": 0.3,
                "low_quality": 0.2
            })
        }
        
        return convert_numpy_types(stats)
    
    def get_processed_files_info(self) -> Dict[str, Any]:
        """Get detailed information about processed files"""
        return convert_numpy_types(self.pdf_processor.get_processed_files_info())
    
    def is_ready(self) -> bool:
        """Check if the RAG pipeline is ready for queries"""
        return self.vector_store.is_collection_available()
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique sources in the vector collection"""
        try:
            return self.vector_store.get_all_sources()
        except Exception as e:
            print(f"Error getting all sources: {str(e)}")
            return []
    
    def delete_documents_by_source(self, source: str) -> Dict[str, Any]:
        """Delete all documents from a specific source"""
        try:
            result = self.vector_store.delete_documents_by_source(source)
            return convert_numpy_types(result)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting documents by source: {str(e)}",
                "documents_deleted": 0
            }