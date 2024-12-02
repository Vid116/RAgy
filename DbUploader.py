import os
import hashlib
from typing import Optional, List, Set, Dict
from pathlib import Path
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    JSONLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()

class DocumentUploader:
    """Handler for uploading multiple document types to a vector database."""
    
    # Map file extensions to appropriate loaders
    LOADER_MAPPING = {
        ".txt": TextLoader,
        ".pdf": PDFMinerLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".json": JSONLoader
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        excluded_dirs: Set[str] = None,
        excluded_files: Set[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.excluded_dirs = excluded_dirs or {'.git', '__pycache__', 'node_modules', 'venv', '.env'}
        self.excluded_files = excluded_files or {'.DS_Store', 'thumbs.db', '.gitignore'}
        
        # Initialize embeddings with error handling
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=embedding_dimensions,
            max_retries=3
        )

        # Add tracking for processed files
        self.processed_files: Dict[str, dict] = {}
        self.hash_log_path: Optional[Path] = None
        self.db: Optional[Chroma] = None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_processed_files(self, db_directory: str):
        """Load the log of processed files."""
        self.hash_log_path = Path(db_directory) / "processed_files.json"
        if self.hash_log_path.exists():
            with open(self.hash_log_path, 'r') as f:
                self.processed_files = json.load(f)

    def _save_processed_files(self):
        """Save the log of processed files."""
        if self.hash_log_path:
            with open(self.hash_log_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)

    def _should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on its hash and modification time.
        Returns True if file is new or modified.
        """
        current_hash = self._calculate_file_hash(file_path)
        mod_time = os.path.getmtime(file_path)
        
        file_info = self.processed_files.get(file_path)
        
        if file_info is None:
            # New file
            self.processed_files[file_path] = {
                "hash": current_hash,
                "mod_time": mod_time,
                "last_processed": mod_time
            }
            return True
            
        if (current_hash != file_info["hash"] or 
            mod_time > file_info["mod_time"]):
            # File has been modified
            self.processed_files[file_path].update({
                "hash": current_hash,
                "mod_time": mod_time,
                "last_processed": mod_time
            })
            return True
            
        # File unchanged
        print(f"Skipping unchanged file: {file_path}")
        return False

    def _get_loader(self, file_path: str) -> Optional[object]:
        """Get the appropriate loader for a file based on its extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_class = self.LOADER_MAPPING.get(file_extension)
        
        if loader_class is None:
            supported_formats = ", ".join(self.LOADER_MAPPING.keys())
            print(f"Unsupported file format: {file_extension}. Supported formats: {supported_formats}")
            return None

        if file_extension == '.json':
            return loader_class(file_path, jq_schema='.', text_content=False)
            
        return loader_class(file_path)

    def _process_file(self, file_path: str) -> List[Document]:
        """Process a single file."""
        loader = self._get_loader(file_path)
        if not loader:
            return []

        try:
            if file_path.lower().endswith(('.html', '.htm')):
                try:
                    documents = loader.load()
                except Exception as e:
                    print(f"First HTML parser failed, trying alternative method: {e}")
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8') as file:
                        soup = BeautifulSoup(file, 'html.parser')
                        text = soup.get_text(separator='\n', strip=True)
                        documents = [Document(page_content=text, metadata={"source": file_path})]
            else:
                documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " "],
                length_function=len,
                is_separator_regex=False
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source_file"] = file_path
                doc.metadata["file_type"] = os.path.splitext(file_path)[1].lower()
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["directory"] = os.path.dirname(os.path.abspath(file_path))
                
            print(f"Successfully processed {file_path}: {len(split_docs)} chunks created")
            return split_docs 
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

    def _get_all_files(self, directory: str) -> List[str]:
        """Recursively get all supported files from directory and subdirectories."""        
        
        all_files = []
        
        for root, dirs, files in os.walk(directory, topdown=True):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file in self.excluded_files:
                    continue
                    
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                
                if file_extension in self.LOADER_MAPPING:
                    all_files.append(file_path)
                    
        return all_files

    def _init_or_load_db(self, db_directory: str, collection_name: str) -> Chroma:
        """Initialize or load existing Chroma database."""
        try:
            db = Chroma(
                persist_directory=db_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            print(f"Loaded existing database with {db._collection.count()} documents")
            return db
        except Exception as e:
            print(f"Creating new database: {e}")
            return Chroma(
                persist_directory=db_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )

    def _get_processed_files_in_db(self) -> Set[str]:
        """Get set of file paths already in database."""
        if not self.db:
            return set()
            
        try:
            # Get all documents' metadata
            results = self.db.get()
            if not results or 'metadatas' not in results:
                return set()
                
            # Extract unique source files
            return {
                meta.get('source_file') 
                for meta in results['metadatas'] 
                if meta and 'source_file' in meta
            }
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return set()

    def upload_directory(
        self,
        directory: str,
        db_directory: str,
        collection_name: str,
        recursive: bool = True,
        force_reload: bool = False
    ) -> Optional[Chroma]:
        """Upload documents from directory to vector database."""
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory")
            return None

        # Initialize or load database
        self.db = self._init_or_load_db(db_directory, collection_name)
        
        # Get files already in database
        processed_files = self._get_processed_files_in_db() if not force_reload else set()
        
        # Get all files in directory
        file_paths = self._get_all_files(directory) if recursive else [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and os.path.splitext(f)[1].lower() in self.LOADER_MAPPING
            and f not in self.excluded_files
        ]
        
        if not file_paths:
            print(f"No supported files found in {directory}")
            return self.db

        # Filter out already processed files
        new_files = [
            path for path in file_paths 
            if path not in processed_files or force_reload
        ]
        
        if not new_files:
            print("No new files to process")
            return self.db

        print(f"Processing {len(new_files)} new files")
        print("Skipping {len(processed_files)} already processed files")
        
        # Process new files
        all_documents = []
        for file_path in new_files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            documents = self._process_file(file_path)
            all_documents.extend(documents)

        if not all_documents:
            print("No documents were successfully processed")
            return self.db

        # Add new documents to database
        self.db.add_documents(documents=all_documents)
        
        print(f"""
        Database updated:
        - New documents processed: {len(all_documents)}
        - Total documents in database: {self.db._collection.count()}
        - Database location: {db_directory}
        - Collection name: {collection_name}
        """)
        
        return self.db

    def upload_documents(
        self,
        file_paths: List[str],
        db_directory: str,
        collection_name: str
    ) -> Optional[Chroma]:
        """Upload multiple documents to a Chroma vector database."""
        try:
            # ... (rest of the method remains the same)
            
            # Get existing database if it exists
            db_path = Path(db_directory)
            if db_path.exists():
                try:
                    existing_db = Chroma(
                        persist_directory=str(db_path),
                        embedding_function=self.embeddings,
                        collection_name=collection_name
                    )
                    print("Found existing database")
                except Exception as e:
                    print(f"Error loading existing database: {e}")
                    existing_db = None
            else:
                existing_db = None
                db_path.mkdir(parents=True, exist_ok=True)

            # Process new documents
            all_documents = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                    
                documents = self._process_file(file_path)
                all_documents.extend(documents)

            if not all_documents:
                print("No documents were successfully processed")
                return existing_db

            # Create or update database
            if existing_db is None:
                db = Chroma.from_documents(
                    documents=all_documents,
                    embedding=self.embeddings,
                    persist_directory=str(db_path),
                    collection_metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 100,
                        "hnsw:search_ef": 50
                    },
                    collection_name=collection_name
                )
            else:
                existing_db.add_documents(documents=all_documents)
                db = existing_db

            db.persist()
            
            print(f"""
            Database updated:
            - New documents processed: {len(all_documents)}
            - Database location: {db_path}
            - Collection name: {collection_name}
            """)
            
            return db
            
        except Exception as e:
            print(f"Fatal error in document upload: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    uploader = DocumentUploader()
    
    # Example with directory processing
    db = uploader.upload_directory(
        directory="C:/Users/Uporabnik/Documents/Asistenca",
        db_directory="./vector_db_MD",
        collection_name="Car_stuff",
        recursive=True,
        force_reload=False
    )