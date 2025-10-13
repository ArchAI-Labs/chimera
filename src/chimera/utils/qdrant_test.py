from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
# Provide your Jina API key and choose the model.
MODEL = "jinaai/jina-embeddings-v2-base-en"
DIMENSIONS = 768  # Set the desired output vector dimensionality.

# Define the inputs
text_input = "A blue cat"

# Get embeddings from the Jina API
from qdrant_client.qdrant_fastembed import TextEmbedding
embedder = TextEmbedding(model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en"))
text_embedding = next(iter(embedder.embed(text_input)))
# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333/")

# Create a collection with named vectors
collection_name = "crew_knowledge"
# client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config={
#         "text_vector": VectorParams(size=DIMENSIONS, distance=Distance.DOT),
#     },
# )
import hashlib, uuid
content_hash = hashlib.sha256(text_input.encode('utf-8')).hexdigest()
deterministic_id = str(uuid.UUID(content_hash[:32]))

# client.upsert(
#     collection_name=collection_name,
#     points=[
#         PointStruct(
#             id=deterministic_id,
#             vector={
#                 "fast-jina-embeddings-v2-base-en": text_embedding,
#             },
#             payload={
#                 'document':text_input,
#                 'source_url':'prova.pippo'
#             }
#         )
#     ],
# )
# print('saved', deterministic_id)
# # Now let's query the collection
search_query = "hyperiot labs"
query_embedding = next(iter(embedder.embed(search_query)))

search_results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    using= 'fast-jina-embeddings-v2-base-en',
    limit=5
).points

for result in search_results:
    print(f"ID: {result.id}, Score: {result.score} Payload: {result.payload}")