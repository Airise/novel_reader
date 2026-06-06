import json
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 允许以 `python indexing/build_vector_store.py` 方式直接运行
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    VECTOR_SIZE,
)

def build_vector_store():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"创建集合 {QDRANT_COLLECTION}")

    # 加载 chunks
    with open(os.path.join(INDEX_DIR, "novel_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)

    points = []
    batch_size = 1000
    for item in tqdm(chunks, desc="向量编码"):
        text = item["text"]
        # 截断过长文本（模型最大长度 512）
        if len(text) > 512:
            text = text[:512]
        vec = model.encode(text, normalize_embeddings=True).tolist()
        points.append(
            PointStruct(
                id=item["id"],
                vector=vec,
                payload={
                    "id": item["id"],
                    "uid": item.get("uid", ""),
                    "text": item["text"],
                    "source": item.get("source", ""),
                },
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            points = []

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print(f"成功上传 {len(chunks)} 个向量点至 {QDRANT_COLLECTION}")

if __name__ == "__main__":
    build_vector_store()