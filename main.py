from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import FileBasedRAG

app = FastAPI()
Rag = FileBasedRAG()

@app.get('/')
def read_root():
    return {'message': 'Hello from Rag system'}

class Document(BaseModel):
    file_path: str
    chunk_size: int = 500
    chunk_overlap: int = 50

@app.post('/add_document')
def load_document(DocumentList: Document):
    try:
        Rag.load_document(DocumentList.file_path, DocumentList.chunk_size, DocumentList.chunk_overlap)
        return {'status': 'success', 'message': DocumentList.file_path + " has successfully loaded into the rag"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to load the file: {e}")

class DirectoryLoad(BaseModel):
    directory: str
    chunk_size: int = 500
    chunk_overlap: int = 50

@app.post('/load_documents_from_directory')
def load_documents_from_directory(DirectoryLoad: DirectoryLoad):
    try:
        Rag.load_documents_from_directory(DirectoryLoad.directory, DirectoryLoad.chunk_size, DirectoryLoad.chunk_overlap)
        return {'status': 'success', 'message': f"Documents from {DirectoryLoad.directory} have been loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to load documents from directory: {e}")

class AddDocs(BaseModel):
    docs: List[str]

@app.post('/add_documents')
def add_documents(AddDocs: AddDocs):
    try:
        Rag.add_documents(AddDocs.docs)
        return {'status': 'success', 'message': f"Added {len(AddDocs.docs)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to add documents: {e}")

@app.post('/build_index')
def build_index():
    try:
        Rag.build_index()
        return {'status': 'success', 'message': 'Index built successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to build index: {e}")

@app.get('/get_document_count')
def get_document_count():
    try:
        count = Rag.get_document_count()
        return {'status': 'success', 'document_count': count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to get document count: {e}")

@app.post('/clear_documents')
def clear_documents():
    try:
        Rag.clear_documents()
        return {'status': 'success', 'message': 'All documents cleared'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to clear documents: {e}")

@app.post('/clear_conversation')
def clear_conversation():
    try:
        Rag.clear_conversation()
        return {'status': 'success', 'message': 'Conversation history cleared'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to clear conversation: {e}")

class SetPrompt(BaseModel):
    prompt: str

@app.post('/set_system_prompt')
def set_system_prompt(SetPrompt: SetPrompt):
    try:
        Rag.set_system_prompt(SetPrompt.prompt)
        return {'status': 'success', 'message': 'System prompt updated'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to set system prompt: {e}")

class Question(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query/")
def query_model(Question: Question):
    try:
        print(Question.question)
        return StreamingResponse(Rag.query(question=Question.question, top_k=Question.top_k), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error occurred while querying the request: {e}')
