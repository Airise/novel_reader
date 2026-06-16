import json
import os
import sys
from typing import List

import jieba
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# 允许以 `python indexing/build_bm25_index.py` 方式直接运行
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config import ES_ANALYZER, ES_HOST, ES_INDEX, INDEX_DIR


def _load_chunks() -> List[dict]:
    with open(os.path.join(INDEX_DIR, "novel_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise RuntimeError("novel_chunks.json 格式错误，应该是列表")
    return chunks


def build_bm25_index():
    try:
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            raise RuntimeError("Elasticsearch ping 失败")
    except Exception as e:
        raise RuntimeError(f"无法连接 Elasticsearch: {e}") from e

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)

    es.indices.create(
        index=ES_INDEX,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "uid": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "chapter_id": {"type": "integer"},
                    "chapter_title": {"type": "keyword"},
                    "chapter_hash": {"type": "keyword"},
                    "chunk_hash": {"type": "keyword"},
                    "text": {"type": "text", "index": False},
                    "text_jieba": {
                        "type": "text",
                        "analyzer": ES_ANALYZER,
                        "search_analyzer": ES_ANALYZER,
                    },
                }
            },
        },
    )
    print(f"创建索引 {ES_INDEX}，使用 jieba + {ES_ANALYZER} 分词")

    chunks = _load_chunks()

    actions = []
    for item in tqdm(chunks, desc="索引文档"):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        text_jieba = " ".join(jieba.cut_for_search(text))
        actions.append(
            {
                "_index": ES_INDEX,
                "_id": int(item["id"]),
                "_source": {
                    "uid": item.get("uid", ""),
                    "source": item.get("source", ""),
                    "chapter_id": int(item.get("chapter_id", 0) or 0),
                    "chapter_title": item.get("chapter_title", ""),
                    "chapter_hash": item.get("chapter_hash", ""),
                    "chunk_hash": item.get("chunk_hash", ""),
                    "text": text,
                    "text_jieba": text_jieba,
                },
            }
        )
        if len(actions) >= 1000:
            helpers.bulk(es, actions)
            actions = []
    if actions:
        helpers.bulk(es, actions)

    es.indices.refresh(index=ES_INDEX)
    print(f"成功索引 {len(chunks)} 个文档至 {ES_INDEX}")


if __name__ == "__main__":
    build_bm25_index()
