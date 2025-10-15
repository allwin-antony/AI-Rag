# AI-RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, providing document-based question answering capabilities. This system allows you to load documents (PDF and TXT files), build semantic search indices, and query an AI model with relevant context for accurate responses.

## Features

- **Document Loading**: Support for PDF and TXT file formats
- **Intelligent Chunking**: Automatic text chunking with configurable overlap for better retrieval
- **Semantic Search**: Uses SentenceTransformers for embedding generation and similarity search
- **LLM Integration**: Powered by Ollama with streaming responses
- **Conversation History**: Maintains chat history for contextual queries
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Flexible Configuration**: Customizable chunk sizes, system prompts, and model selection

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Pull the required model: `ollama pull mistral` (or your preferred model)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/allwin-antony/AI-Rag.git
cd ai-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Run the FastAPI server:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Example Workflow

1. **Load Documents**:
```bash
# Load a single document
curl -X POST "http://localhost:8000/add_document" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "/path/to/document.pdf", "chunk_size": 500, "chunk_overlap": 50}'

# Load all documents from a directory
curl -X POST "http://localhost:8000/load_documents_from_directory" \
     -H "Content-Type: application/json" \
     -d '{"directory": "/path/to/documents/", "chunk_size": 500, "chunk_overlap": 50}'
```

2. **Build Index** (optional - automatically built on first query):
```bash
curl -X POST "http://localhost:8000/build_index"
```

3. **Query the System**:
```bash
curl -X POST "http://localhost:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the document?", "top_k": 3}'
```

## API Endpoints

### Core Endpoints

- `GET /` - Health check endpoint
- `POST /add_document` - Load a single document
- `POST /load_documents_from_directory` - Load all documents from a directory
- `POST /add_documents` - Add documents directly as text
- `POST /build_index` - Manually build the search index
- `GET /get_document_count` - Get the number of loaded document chunks
- `POST /clear_documents` - Clear all loaded documents
- `POST /clear_conversation` - Clear conversation history
- `POST /set_system_prompt` - Update the system prompt
- `POST /query/` - Query the RAG system (streaming response)

### Request/Response Examples

#### Add Document
```json
// Request
{
  "file_path": "/path/to/document.pdf",
  "chunk_size": 500,
  "chunk_overlap": 50
}

// Response
{
  "status": "success",
  "message": "/path/to/document.pdf has successfully loaded into the rag"
}
```

#### Query
```json
// Request
{
  "question": "Summarize the key points",
  "top_k": 3
}

// Response (streaming)
"Based on the provided documents, the key points are..."
```

## Configuration

### Model Configuration

The system uses Ollama for LLM inference. You can change the model by modifying the `model_name` parameter in the `FileBasedRAG` class initialization (default: "mistral").

### Chunking Parameters

- `chunk_size`: Number of words per chunk (default: 500)
- `chunk_overlap`: Number of overlapping words between chunks (default: 50)

### System Prompt

Customize the AI behavior by setting a custom system prompt:
```bash
curl -X POST "http://localhost:8000/set_system_prompt" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "You are a technical documentation assistant..."}'
```

## Supported File Formats

- **PDF**: Requires PyPDF2 (included in requirements.txt)
- **TXT**: Plain text files with UTF-8 encoding

## Dependencies

Key dependencies include:
- `fastapi`: Web framework
- `ollama`: LLM inference
- `sentence-transformers`: Text embeddings
- `PyPDF2`: PDF text extraction
- `numpy`: Numerical computations
- `uvicorn`: ASGI server

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running locally and the specified model is pulled
- **PDF Loading Errors**: Verify PyPDF2 is installed and PDFs are not password-protected
- **Memory Issues**: For large document collections, consider adjusting chunk sizes or using a more powerful machine
