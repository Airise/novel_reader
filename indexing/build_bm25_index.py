import json
import os
import sys

import jieba
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# 允许以 `python indexing/build_bm25_index.py` 方式直接运行
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config import ES_ANALYZER, ES_HOST, ES_INDEX, INDEX_DIR


def build_bm25_index():
    try:
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            raise RuntimeError("Elasticsearch ping 失败")
    except Exception as e:
        raise RuntimeError(f"无法连接 Elasticsearch: {e}") from e

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)

    # 创建索引：text 为原文，text_jieba 为 jieba 预分词后的空格串
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

    # 加载 chunks
    with open(os.path.join(INDEX_DIR, "novel_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    actions = []
    for item in tqdm(chunks, desc="索引文档"):
        text = item["text"]
        text_jieba = " ".join(jieba.cut_for_search(text))
        actions.append(
            {
                "_index": ES_INDEX,
                "_id": item["id"],
                "_source": {
                    "uid": item.get("uid", ""),
                    "source": item.get("source", ""),
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