from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from services.rag_engine import RAGEngine
from utils.helpers import save_uploaded_file, validate_pdf_file, get_file_size_mb
from config.settings import settings
from pathlib import Path
import os

router = APIRouter()
rag_engine = RAGEngine()

# Request models
class ResetRequest(BaseModel):
    include_pdfs: bool = True
    confirm: bool = False

@router.get("/")
async def root():
    return {"message": "Research Paper RAG API", "status": "running"}

@router.get("/health")
async def health_check():
    stats = rag_engine.get_system_stats()
    return {
        "status": "healthy",
        "system_stats": stats
    }

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload PDF files"""
    try:
        uploaded_files = []
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Save file
            file_path = os.path.join(settings.PDF_STORAGE_PATH, file.filename)
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Validate PDF
            if not validate_pdf_file(file_path):
                os.remove(file_path)
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid PDF")
            
            uploaded_files.append({
                "filename": file.filename,
                "size_mb": round(get_file_size_mb(file_path), 2),
                "path": file_path
            })
        
        return JSONResponse({
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_documents():
    """Ingest all PDF documents in the storage directory"""
    try:
        result = rag_engine.ingest_documents(pdf_directory=settings.PDF_STORAGE_PATH)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(
    query: str = Form(...), 
    k: int = Form(5),
    threshold: Optional[float] = Form(None)  # Changed from Form(0.5) to Form(None)
):
    """Search for relevant documents with optional similarity threshold"""
    try:
        results = rag_engine.search_documents(query=query, k=k, threshold=threshold)
        return JSONResponse({
            "query": query,
            "results": results,
            "total_results": len(results),
            "threshold_used": threshold,
            "threshold_applied": threshold is not None,
            "search_method": "cosine_similarity"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask")
async def ask_question(
    question: str = Form(...), 
    k: int = Form(5),
    threshold: Optional[float] = Form(None)  # Already correct - keep as None
):
    """Ask a question and get an answer using RAG with optional similarity threshold"""
    try:
        result = rag_engine.generate_answer(query=question, k=k, threshold=threshold)
        return JSONResponse({
            "question": question,
            "threshold_applied": threshold is not None,
            "threshold_value": threshold,
            "threshold_mode": "filtered" if threshold is not None else "unfiltered",
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def summarize_documents(
    query: Optional[str] = Form(None), 
    k: int = Form(10),
    threshold: Optional[float] = Form(None)  # Already correct - keep as None
):
    """Generate summary of documents with optional similarity threshold"""
    try:
        result = rag_engine.summarize_documents(query=query, k=k, threshold=threshold)
        return JSONResponse({
            "query": query,
            "threshold_applied": threshold is not None,
            "threshold_value": threshold,
            "threshold_mode": "filtered" if threshold is not None else "unfiltered",
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/sources")
async def get_all_sources():
    """Get list of all unique sources in the vector collection"""
    try:
        sources = rag_engine.get_all_sources()
        return JSONResponse({
            "sources": sources,
            "total_sources": len(sources)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/source")
async def search_by_source(source: str = Form(...)):
    """Search documents by source file name"""
    try:
        results = rag_engine.search_by_source(source)
        return JSONResponse({
            "source": source,
            "results": results,
            "total_results": len(results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sources/{source_name}")
async def delete_documents_by_source(source_name: str):
    """Delete all documents from a specific source"""
    try:
        result = rag_engine.delete_documents_by_source(source_name)
        return JSONResponse({
            "message": f"Documents deletion completed for source: {source_name}",
            "details": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = rag_engine.get_system_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processed-files")
async def get_processed_files():
    """Get detailed information about processed files"""
    try:
        files_info = rag_engine.get_processed_files_info()
        return JSONResponse({
            "processed_files": files_info,
            "total_files": len(files_info)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ready")
async def check_system_ready():
    """Check if the RAG system is ready for queries"""
    try:
        is_ready = rag_engine.is_ready()
        return JSONResponse({
            "ready": is_ready,
            "message": "System is ready for queries" if is_ready else "System is not ready - please ingest documents first"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset")
async def reset_system(include_pdfs: bool = Form(True)):
    """Reset the entire system (clear all data)"""
    try:
        result = rag_engine.reset_system(include_pdfs=include_pdfs)
        return JSONResponse({
            "message": "System reset completed",
            "details": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset/vector-store")
async def clear_vector_store():
    """Clear only the vector store, keeping PDF files"""
    try:
        result = rag_engine.clear_vector_store_only()
        return JSONResponse({
            "message": "Vector store cleared successfully",
            "details": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset/pdf-storage")
async def clear_pdf_storage():
    """Clear only PDF storage files"""
    try:
        result = rag_engine.clear_pdf_storage()
        return JSONResponse({
            "message": "PDF storage cleared successfully",
            "details": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset/confirm")
async def reset_system_with_confirmation(reset_request: ResetRequest):
    """Reset system with explicit confirmation"""
    try:
        if not reset_request.confirm:
            raise HTTPException(
                status_code=400, 
                detail="Reset confirmation required. Set 'confirm' to true to proceed."
            )
        
        result = rag_engine.reset_system(include_pdfs=reset_request.include_pdfs)
        return JSONResponse({
            "message": "System reset completed with confirmation",
            "details": result,
            "settings": {
                "include_pdfs": reset_request.include_pdfs,
                "confirmed": reset_request.confirm
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))