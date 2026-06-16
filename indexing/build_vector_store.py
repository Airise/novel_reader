import json
import os
import sys
from typing import List

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


def _load_chunks() -> List[dict]:
    with open(os.path.join(INDEX_DIR, "novel_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise RuntimeError("novel_chunks.json 格式错误，应该是列表")
    return chunks


def _normalize_text_for_embedding(text: str) -> str:
    text = (text or "").strip()
    if len(text) > 512:
        text = text[:512]
    return text


def build_vector_store():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"创建集合 {QDRANT_COLLECTION}")

    chunks = _load_chunks()
    model = SentenceTransformer(EMBEDDING_MODEL)

    points = []
    batch_size = 1000
    for item in tqdm(chunks, desc="向量编码"):
        text = _normalize_text_for_embedding(item.get("text", ""))
        if not text:
            continue

        vec = model.encode(text, normalize_embeddings=True).tolist()
        points.append(
            PointStruct(
                id=int(item["id"]),
                vector=vec,
                payload={
                    "id": int(item["id"]),
                    "uid": item.get("uid", ""),
                    "text": item.get("text", ""),
                    "source": item.get("source", ""),
                    "chapter_id": int(item.get("chapter_id", 0) or 0),
                    "chapter_title": item.get("chapter_title", ""),
                    "chapter_hash": item.get("chapter_hash", ""),
                    "chunk_hash": item.get("chunk_hash", ""),
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
