import json
import numpy as np
import os

# Supprimer la ligne qui force le CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # doit être AVANT tout import torch

import torch
from sentence_transformers import SentenceTransformer

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
print(f"{len(all_chunks)} chunks chargés pour embedding")

# Détecter automatiquement le device disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device détecté : {device}")

# Option : utiliser un modèle plus léger pour accélérer
# embedding_model = SentenceTransformer("intfloat/e5-base-v2", device=device)  # Plus rapide que e5-large
embedding_model = SentenceTransformer("intfloat/e5-large", device=device)  # Garder large pour qualité

chunk_embeddings = embedding_model.encode(
    all_chunks,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"Dimension d'un embedding : {chunk_embeddings[0].shape}")

np.save("data/processed/chunk_embeddings.npy", chunk_embeddings)
print("Embeddings sauvegardés dans data/processed/chunk_embeddings.npy")