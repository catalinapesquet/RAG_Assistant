import os
import json
import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:"
    "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

connections.connect(alias="default", host="172.17.0.1", port=19530)
collection = Collection("documents")
collection.load()

# Détecter automatiquement le device pour l'embedding
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device pour embedding : {device}")

embedding_model=SentenceTransformer("intfloat/e5-large", device=device)

# Augmenter les GPU layers pour accélérer la génération
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=35,  # augmenté de 25 à 35
    n_ctx=4096,
    verbose=False
)

# Cache simple pour les embeddings des questions
question_cache = {}

def retrieve(question, top_k=3):
    """Search for relevant chunks in Milvus and return their texts."""
    # Vérifier le cache
    if question in question_cache:
        q_vec = question_cache[question]
    else:
        q_vec = embedding_model.encode(
          [f"query: {question}"],
          normalize_embeddings=True  
        )[0].tolist()
        question_cache[question] = q_vec  # Ajouter au cache
    
    results = collection.search(
        data=[q_vec], # vector to search
        anns_field="embedding", # name of the vector field
        param={"metric_type": "IP", "params": {"nprobe": 10}}, # search parameters
        limit=top_k, # number of results to return
        output_fields=["text"] # fields to return
    )

    chunks = []
    print("\n Chunks retrieved:")
    for i, hit in enumerate(results[0]):
        text = hit.entity.get("text")
        print(f"[{i+1}] Score: {hit.score:.4f} | {text[:80]}...")
        chunks.append(text)

    return chunks

def generate(question, chunks):
    """Generate an answer using the LLM based on the question and retrieved chunks."""
    context = "\n\n".join(chunks)
    if len(context) > 3000:
        context = context[:3000]

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    print(f"DEBUG: Contexte size = {len(context)} chars")  
    print(f"DEBUG: Prompt length = {len(prompt)} chars")
    
    response = llm(prompt, max_tokens=512, temperature=0.5, stop=["</s>", "[INST]", "Question:"])
    return response['choices'][0]['text'].strip()

def rag_query(question):
    chunks = retrieve(question)
    answer = generate(question, chunks)
    return answer

if __name__ == "__main__":
    print("RAG ready. Type 'q' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == 'q':
            break
        if not question:
            continue
        answer = rag_query(question)
        print(f"\nAnswer: {answer}\n")
        print("─" * 60)