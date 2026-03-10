# RAG Assistant - Intelligent Notification System

A locally-hosted retrieval-augmented generation system for intelligent notification management, combining vector search with large language models.

## Overview

This system implements a complete RAG (Retrieval-Augmented Generation) pipeline for intelligent notification management:

1. **Document Processing** - Extract and chunk notification-related texts from PDFs
2. **Vector Indexing** - Generate embeddings and store in Milvus vector database
3. **Question Answering** - Retrieve relevant context and generate answers using Mistral-7B

## Features

- **Multilingual Support** - Handles 95+ languages via e5-large embeddings
- **Local LLM** - Mistral-7B running via llama.cpp (no API keys required)
- **Vector Database** - Milvus for efficient similarity search
- **Web Interface** - FastAPI-based UI for interactive Q&A
- **GPU Acceleration** - CUDA support for faster inference

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 4GB+ GPU VRAM (for LLM inference)

## Quick Start

### 1. Start Services

```bash
cd llm
docker-compose up -d
```

This launches:
- Milvus vector database (port 19530)
- MinIO object storage (port 9000)
- etcd key-value store

### 2. Process Documents

```bash
# Extract text from PDFs
python ingest.py

# Generate embeddings
python embed.py

# Index in Milvus
python indexing.py
```

### 3. Launch RAG System

**Command Line:**
```bash
python rag.py
```

**API Server:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Web UI:** Navigate to `http://localhost:8000`

## API Usage

### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do intelligent notifications work?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Intelligent notifications use machine learning...",
  "sources": [
    {"text": "Intelligent notifications analyze...", "score": 0.92}
  ],
  "elapsed_ms": 8450
}
```

## Configuration

### Model Settings (rag.py)

```python
# Embedding model
embedding_model = SentenceTransformer("intfloat/e5-large", device="cuda")

# LLM configuration
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=35,        # GPU layers (adjust for VRAM)
    n_ctx=4096,             # Context window
    verbose=False
)

# Generation parameters
max_tokens=512             # Maximum response length
temperature=0.5            # Response creativity
```

### Environment Variables

Create `.env`:
```env
MILVUS_HOST=172.17.0.1
MILVUS_PORT=19530
API_HOST=0.0.0.0
API_PORT=8000
```

## Project Structure

```
llm/
├── rag.py              # Main RAG pipeline
├── api.py              # FastAPI server
├── embed.py            # Embedding generation
├── ingest.py           # PDF processing
├── indexing.py         # Milvus indexing
├── retriever.py        # Retrieval utilities
├── db.py               # Database operations
├── test.py             # Testing scripts
├── index.html          # Web UI
├── docker-compose.yml  # Services
├── Dockerfile          # Container build
└── models/             # LLM checkpoints
```

## Dependencies

Key packages:
- `sentence-transformers` - Text embeddings
- `llama-cpp-python` - LLM inference
- `pymilvus` - Vector database
- `fastapi` - Web API
- `pdfplumber` - PDF processing

**Version**: 1.0  
**Last Updated**: March 2026
