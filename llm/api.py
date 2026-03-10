"""
api.py — FastAPI wrapper around rag.py
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import os

# ── Import your RAG module ──────────────────────────────────────────────────
# Make sure rag.py is in the same directory (or adjust the import path)
from rag import rag_query, retrieve

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (place index.html next to api.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Schemas ────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    text: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    elapsed_ms: int


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    """Serve the HTML interface."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run a RAG query and return the answer + source chunks."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.monotonic()

    # Retrieve source chunks (with scores if your retrieve() supports it)
    try:
        chunks = retrieve(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    # Generate answer
    try:
        answer = rag_query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    elapsed = int((time.monotonic() - t0) * 1000)

    sources = [SourceChunk(text=c if isinstance(c, str) else c.get("text", str(c)))
               for c in chunks]

    return QueryResponse(answer=answer, sources=sources, elapsed_ms=elapsed)


@app.get("/health")
def health():
    return {"status": "ok"}