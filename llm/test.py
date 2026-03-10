from pymilvus import Collection
from pymilvus import connections

connections.connect(
    alias="default",
    host="172.17.0.1",  # default Docker bridge gateway
    port=19530
)

# Si tu es déjà connecté, pas besoin de reconnect
collection = Collection("documents")
collection.load()
print("Nombre de documents:", collection.num_entities)

for field in collection.schema.fields:
    print(f"Champ: {field.name}, Type: {field.dtype}, Params: {field.params}")

results = collection.query(
    expr="id >= 0",
    output_fields=["id", "text"],
    limit=10
)
for r in results:
    print(f"ID {r['id']}: {r['text'][:150]}")
    print("---")

collection.load()
from sentence_transformers import SentenceTransformer

query_model = SentenceTransformer("intfloat/e5-large")  
query_text = "user's attention"
query_embedding = query_model.encode([query_text], normalize_embeddings=True).tolist()

results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["text"]
)

for hits in results:
    for hit in hits:
        print(f"Score: {hit.score:.4f}")
        print(f"Texte: {hit.entity.get('text')[:300]}")
        print("---")