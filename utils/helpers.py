import os
import shutil
from pathlib import Path
from typing import List
import tempfile

def save_uploaded_file(uploaded_file, destination_path: str) -> str:
    """Save uploaded file to destination"""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    with open(destination_path, "wb") as f:
        if hasattr(uploaded_file, 'read'):
            f.write(uploaded_file.read())
        else:
            f.write(uploaded_file)
    
    return destination_path

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def validate_pdf_file(file_path: str) -> bool:
    """Validate if file is a valid PDF"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except:
        return False

def clean_directory(directory_path: str, keep_files: List[str] = None):
    """Clean directory but keep specified files"""
    if not os.path.exists(directory_path):
        return
    
    keep_files = keep_files or []
    
    for filename in os.listdir(directory_path):
        if filename not in keep_files:
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")