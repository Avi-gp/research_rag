from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from services.rag_pipeline import RAGPipeline
from utils.helpers import save_uploaded_file, validate_pdf_file, get_file_size_mb
from config.settings import settings
from pathlib import Path
import os

router = APIRouter()
rag_pipeline = RAGPipeline()

# Request models
class ResetRequest(BaseModel):
    include_pdfs: bool = True
    confirm: bool = False

@router.get("/")
async def root():
    return {"message": "Research Paper RAG API", "status": "running"}

@router.get("/health")
async def health_check():
    stats = rag_pipeline.get_system_stats()
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
        result = rag_pipeline.ingest_documents(pdf_directory=settings.PDF_STORAGE_PATH)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(query: str = Form(...), k: int = Form(5)):
    """Search for relevant documents"""
    try:
        results = rag_pipeline.search_documents(query, k)
        return JSONResponse({
            "query": query,
            "results": results,
            "total_results": len(results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask")
async def ask_question(question: str = Form(...), k: int = Form(5)):
    """Ask a question and get an answer using RAG"""
    try:
        result = rag_pipeline.generate_answer(question, k)
        return JSONResponse({
            "question": question,
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def summarize_documents(query: Optional[str] = Form(None), k: int = Form(10)):
    """Generate summary of documents"""
    try:
        result = rag_pipeline.summarize_documents(query, k)
        return JSONResponse({
            "query": query,
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = rag_pipeline.get_system_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset")
async def reset_system(include_pdfs: bool = Form(True)):
    """Reset the entire system (clear all data)"""
    try:
        result = rag_pipeline.reset_system(include_pdfs=include_pdfs)
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
        result = rag_pipeline.clear_vector_store_only()
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
        result = rag_pipeline.clear_pdf_storage()
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
        
        result = rag_pipeline.reset_system(include_pdfs=reset_request.include_pdfs)
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