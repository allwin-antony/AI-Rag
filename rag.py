import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Generator
from pathlib import Path
import os


try:
    import PyPDF2
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False
    print("Warning: PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2")

class FileBasedRAG:
    def __init__(self, model_name: str = "mistral",max_history: int = 10):
        """Initialize RAG system with automatic index management"""
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self._documents_dirty = False 
        self.max_history = max_history
        self.chat_messages: List[Dict[str, str]] = [] 
        self.system_prompt = """You are a helpful assistant that answers questions based on provided context. 
            Use the relevant documents and conversation history to provide accurate answers. 
            If you don't know something based on the provided context, say so clearly."""
    def _read_pdf_file(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_SUPPORTED:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
        return text.strip()
    
    def _read_text_file(self, file_path: str) -> str:
        """Read text from .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - chunk_overlap if end < len(words) else end
            
        return chunks
    
    def _mark_dirty(self):
        """Mark that documents have changed and embeddings need updating"""
        self._documents_dirty = True
    
    def _ensure_index_built(self):
        """Build index if documents have changed or no index exists"""
        if self._documents_dirty or self.embeddings is None:
            if not self.documents:
                print("Warning: No documents loaded. Query will use direct model call.")
                self.embeddings = None
                self._documents_dirty = False
                return
            
            print(f"Building index for {len(self.documents)} document chunks...")
            self.embeddings = self.embedding_model.encode(self.documents)
            self._documents_dirty = False
            print("Index built successfully!")

    def _get_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k most relevant document chunks"""
        # Ensure index is up to date
        self._ensure_index_built()
        
        if self.embeddings is None:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def _build_chat_messages(self, question: str, relevant_docs: List[str]) -> List[Dict[str, str]]:
        """Build complete chat message history with context"""
        messages = []
        
        # System message with instructions
        system_content = self.system_prompt
        if relevant_docs:
            doc_context = "Relevant documents:\n" + "\n\n".join(relevant_docs)
            system_content += f"\n\n{doc_context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        messages.extend(self.chat_messages[-self.max_history:])
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        return messages
  
    def load_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Load a single document (PDF or TXT) and mark index as dirty"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file based on extension
        if file_path.suffix.lower() == '.pdf':
            text = self._read_pdf_file(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            text = self._read_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported: .pdf, .txt")
        
        if not text:
            print(f"Warning: No text extracted from {file_path}")
            return
        
        # Chunk the text
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        self.documents.extend(chunks)
        self._mark_dirty()  # Mark that we need to rebuild index
        print(f"Loaded {len(chunks)} chunks from {file_path.name}")
    
    def load_documents_from_directory(self, directory: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Load all PDF and TXT files from a directory and mark index as dirty"""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        supported_extensions = {'.pdf', '.txt'}
        files_processed = 0
        new_chunks = []
        
        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in supported_extensions and file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        text = self._read_pdf_file(str(file_path))
                    else:
                        text = self._read_text_file(str(file_path))
                    
                    if text:
                        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                        new_chunks.extend(chunks)
                        files_processed += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if new_chunks:
            self.documents.extend(new_chunks)
            self._mark_dirty()
        
        print(f"Processed {files_processed} files from {directory}")
    
    def add_documents(self, docs: List[str]):
        """Add documents directly and mark index as dirty"""
        if docs:
            self.documents.extend(docs)
            self._mark_dirty()
            print(f"Added {len(docs)} documents directly")
    
    def build_index(self):
        """Force rebuild the embedding index (useful for manual control)"""
        self._ensure_index_built()
    
    def query(self, question: str, top_k: int = 3) -> Generator[str, None, None]:
        """Query the RAG system """
        # Get relevant documents
        relevant_docs = self._get_relevant_docs(question, top_k)
        
        # Build chat messages
        messages = self._build_chat_messages(question, relevant_docs)

        full_response = ""
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                full_response += token
                yield token
                
        except Exception as e:
            error_msg = f"Error during streaming: {e}"
            yield error_msg
            full_response = error_msg
        
        # Add message to History
        self.chat_messages.append({"role": "user", "content": question})
        self.chat_messages.append({"role": "assistant", "content": full_response})
        
        # Trim history if too long
        if len(self.chat_messages) > self.max_history * 2:  
            self.chat_messages = self.chat_messages[-(self.max_history * 2):]
        
        return
    
    def get_document_count(self) -> int:
        """Get the current number of document chunks"""
        return len(self.documents)
    
    def clear_documents(self):
        """Clear all documents and reset the index"""
        self.documents = []
        self.embeddings = None
        self._documents_dirty = False
        print("All documents cleared.")
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.chat_messages = []
    
    def set_system_prompt(self, prompt: str):
        """Customize system prompt"""
        self.system_prompt = prompt