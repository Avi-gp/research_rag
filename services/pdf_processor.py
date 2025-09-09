from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from config.settings import settings
import hashlib
import json
import os

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.processed_files_path = Path(settings.VECTOR_DB_PATH) / "processed_files.json"
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> dict:
        """Load processed files metadata"""
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_processed_files(self):
        """Save processed files metadata"""
        os.makedirs(os.path.dirname(self.processed_files_path), exist_ok=True)
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Warning: Could not generate hash for {file_path}: {e}")
            return ""
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if file needs processing based on hash comparison"""
        file_name = Path(file_path).name
        current_hash = self._get_file_hash(file_path)
        
        if not current_hash:
            return True  # Process if we can't get hash
            
        if file_name not in self.processed_files:
            return True  # Process if never processed before
            
        return self.processed_files[file_name].get('hash') != current_hash
    
    def _update_processed_file_record(self, file_path: str, chunks_count: int):
        """Update the record for a processed file"""
        file_name = Path(file_path).name
        file_hash = self._get_file_hash(file_path)
        
        self.processed_files[file_name] = {
            "hash": file_hash,
            "chunks": chunks_count,
            "processed_at": str(Path(file_path).stat().st_mtime),
            "file_path": file_path
        }
    
    def process_pdfs_from_directory(self, pdf_directory: str) -> List[Document]:
        """
        Process all PDFs in a directory using LangChain's DirectoryLoader and PyMuPDFLoader.
        Only processes files that are new or have changed.
        """
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Directory {pdf_directory} does not exist")
        
        # Get all PDF files in directory
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return []
        
        # Filter files that need processing
        files_to_process = [
            str(pdf_file) for pdf_file in pdf_files 
            if self._should_process_file(str(pdf_file))
        ]
        
        if not files_to_process:
            print("All PDF files are already processed and up to date.")
            return []
        
        print(f"Processing {len(files_to_process)} PDF files...")
        
        try:
            # Create DirectoryLoader with PyMuPDFLoader
            loader = DirectoryLoader(
                pdf_directory,
                glob="*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            
            # Load documents
            documents = loader.load()
            
            if not documents:
                print("No documents were loaded from the directory")
                return []
            
            # Filter documents to only include those that need processing
            filtered_documents = [
                doc for doc in documents 
                if doc.metadata.get('source') in files_to_process
            ]
            
            print(f"Loaded {len(filtered_documents)} documents from {len(files_to_process)} files")
            
            # Split documents into chunks
            all_chunks = self.text_splitter.split_documents(filtered_documents)
            
            # Add chunk metadata and update processed files record
            chunks_by_source = {}
            for i, chunk in enumerate(all_chunks):
                source = chunk.metadata.get('source', 'unknown')
                
                # Count chunks per source
                if source not in chunks_by_source:
                    chunks_by_source[source] = 0
                chunks_by_source[source] += 1
                
                # Add chunk metadata
                chunk.metadata.update({
                    "chunk_id": chunks_by_source[source] - 1,
                    "global_chunk_id": i
                })
            
            # Update total chunks for each source and save records
            for chunk in all_chunks:
                source = chunk.metadata.get('source')
                if source:
                    chunk.metadata["total_chunks"] = chunks_by_source.get(source, 1)
                    
            # Update processed files records
            for source_file in chunks_by_source:
                self._update_processed_file_record(source_file, chunks_by_source[source_file])
                print(f"Processed {Path(source_file).name}: {chunks_by_source[source_file]} chunks")
            
            # Save processed files metadata
            self._save_processed_files()
            
            print(f"Total chunks created: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            raise Exception(f"Error processing PDFs from directory: {str(e)}")
    
    def get_processed_files_info(self) -> dict:
        """Get information about processed files"""
        return self.processed_files.copy()