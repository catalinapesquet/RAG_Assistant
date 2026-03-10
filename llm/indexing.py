import numpy as np
import os
import json
from pymilvus import utility
from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection


connections.connect(
    alias="default",
    host="172.17.0.1",  # default Docker bridge gateway
    port=19530
)

if utility.has_collection("documents"):
    Collection("documents").drop()

embeddings = np.load("data/processed/chunk_embeddings.npy")
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ID
id_field = FieldSchema(
    name="id", 
    dtype=DataType.INT64,
    is_primary=True,
)

# Vector embedding
embedding_field = FieldSchema(
    name="embedding", 
    dtype=DataType.FLOAT_VECTOR,
    dim=1024
)

# Texte
text_field = FieldSchema(
    name='text',
    dtype=DataType.VARCHAR,
    max_length=16384
)

# Construct schema
schema = CollectionSchema(
    fields=[id_field, embedding_field, text_field],
    description="Collection for embeddings"
)

# Create collection
collection = Collection(
    name='documents',
    schema=schema, 
    using="default"
)

collection.insert([
    list(range(len(chunks))),
    embeddings.tolist(),
    chunks
])


collection.flush()

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "IP",
        "index_type": "HNSW",  # Changé de IVF_FLAT à HNSW pour plus de vitesse
        "params": {"M": 16, "efConstruction": 256}
    }
)

print("Indexing done.")