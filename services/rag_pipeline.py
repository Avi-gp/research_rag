import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.llm_service import LLMService
from config.settings import settings

class RAGPipeline:
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
                self.vector_store.add_documents(documents)
                
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
    
    def search_documents(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for relevant documents using similarity search with threshold"""
        try:
            return self.vector_store.get_relevant_documents(query=query, k=k, threshold=threshold)
        except Exception as e:
            print(f"Error in search_documents: {str(e)}")
            return []
    
    def get_retriever(self, k: int = 5, threshold: float = 0.5):
        """Get a LangChain retriever for use with chains"""
        try:
            return self.vector_store.as_retriever(k=k, threshold=threshold)
        except Exception as e:
            print(f"Error getting retriever: {str(e)}")
            return None
    
    def generate_answer(self, query: str, k: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
        """Generate answer using RAG pipeline with similarity search and threshold"""
        try:
            # Search for relevant documents using similarity search with threshold
            relevant_docs = self.search_documents(query=query, k=k, threshold=threshold)
            
            if not relevant_docs:
                return {
                    "answer": f"I couldn't find any relevant information in the knowledge base above the similarity threshold of {threshold}. Try lowering the threshold or check if documents have been ingested.",
                    "sources": [],
                    "context_used": 0,
                    "threshold_used": threshold
                }
            
            # Combine context from relevant documents
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Generate answer
            answer = self.llm_service.answer_question(context, query)
            
            # Extract source information
            sources = []
            for doc in relevant_docs:
                source_info = {
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "chunk_id": doc['metadata'].get('chunk_id', 0),
                    "rank": doc.get('rank', 0),
                    "similarity_score": doc.get('similarity_score', 0.0)
                }
                
                if source_info not in sources:
                    sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(relevant_docs),
                "threshold_used": threshold
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "context_used": 0,
                "threshold_used": threshold
            }
    
    def summarize_documents(self, query: str = None, k: int = 10, threshold: float = 0.4) -> Dict[str, Any]:
        """Generate summary of documents using similarity search with threshold"""
        try:
            if query:
                # Search for relevant documents based on query
                relevant_docs = self.search_documents(query=query, k=k, threshold=threshold)
            else:
                # Get documents for general summary with lower threshold
                relevant_docs = self.search_documents(query="research summary overview", k=k, threshold=threshold)
            
            if not relevant_docs:
                return {
                    "summary": f"No documents found for summarization above the similarity threshold of {threshold}.",
                    "sources": [],
                    "context_used": 0,
                    "threshold_used": threshold
                }
            
            # Combine context
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Generate summary
            summary = self.llm_service.generate_summary(context, query)
            
            # Extract sources with counts and similarity scores
            sources = {}
            for doc in relevant_docs:
                source = doc['metadata'].get('source', 'Unknown')
                if source in sources:
                    sources[source]['chunks'] += 1
                    sources[source]['total_similarity'] += doc.get('similarity_score', 0.0)
                    sources[source]['max_similarity'] = max(
                        sources[source]['max_similarity'], 
                        doc.get('similarity_score', 0.0)
                    )
                else:
                    sources[source] = {
                        'chunks': 1,
                        'total_similarity': doc.get('similarity_score', 0.0),
                        'max_similarity': doc.get('similarity_score', 0.0)
                    }
            
            # Convert to list format with similarity metrics
            source_list = []
            for source, info in sources.items():
                source_info = {
                    "source": source,
                    "chunks_used": info['chunks'],
                    "avg_similarity": info['total_similarity'] / info['chunks'] if info['chunks'] > 0 else 0.0,
                    "max_similarity": info['max_similarity']
                }
                source_list.append(source_info)
            
            return {
                "summary": summary,
                "sources": source_list,
                "context_used": len(relevant_docs),
                "threshold_used": threshold
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "sources": [],
                "context_used": 0,
                "threshold_used": threshold
            }
    
    def search_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Search documents by source file name"""
        try:
            documents = self.vector_store.search_by_source(source_name)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get('chunk_id', 0)
                }
                for doc in documents
            ]
        except Exception as e:
            print(f"Error searching by source: {str(e)}")
            return []
    
    def analyze_query_similarity(self, query: str, k: int = 50, threshold: float = 0.0) -> Dict[str, Any]:
        """Analyze similarity scores for a query across documents"""
        try:
            # Get similarity search results with scores (use low threshold to get all)
            results = self.vector_store.similarity_search(query, k=k, threshold=threshold)
            
            if len(results) == 0:
                return {
                    "status": "error",
                    "message": "No documents available for analysis"
                }
            
            similarities = [result.get('similarity_score', 0.0) for result in results]
            
            if not similarities:
                return {
                    "status": "error",
                    "message": "No similarity scores available for analysis"
                }
            
            import numpy as np
            
            return {
                "status": "success",
                "total_documents": len(similarities),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "above_threshold": {
                    "0.9": int(np.sum(np.array(similarities) >= 0.9)),
                    "0.8": int(np.sum(np.array(similarities) >= 0.8)),
                    "0.7": int(np.sum(np.array(similarities) >= 0.7)),
                    "0.6": int(np.sum(np.array(similarities) >= 0.6)),
                    "0.5": int(np.sum(np.array(similarities) >= 0.5)),
                    "0.4": int(np.sum(np.array(similarities) >= 0.4)),
                    "0.3": int(np.sum(np.array(similarities) >= 0.3))
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing similarities: {str(e)}"
            }
    
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
            
            # Reset vector store
            vector_result = self.vector_store.clear_store()
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
            
            return results
            
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
            result = self.vector_store.clear_store()
            
            # Reset PDF processor processed files tracking so files will be reprocessed
            self.pdf_processor.processed_files = {}
            self.pdf_processor._save_processed_files()
            
            return {
                "status": result["status"],
                "message": f"{result['message']} and reset processing tracking",
                "documents_cleared": result.get("documents_cleared", 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing vector store: {str(e)}",
                "documents_cleared": 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_stats()
        
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
        
        return {
            "vector_store": vector_stats,
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
                "search_method": "similarity_score_threshold",
                "default_threshold": 0.5,
                "langchain_retriever_support": True,
                "similarity_analysis": True,
                "source_based_search": True,
                "incremental_processing": True
            }
        }
    
    def get_processed_files_info(self) -> Dict[str, Any]:
        """Get detailed information about processed files"""
        return self.pdf_processor.get_processed_files_info()