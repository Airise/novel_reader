import time

import jieba
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from retrieval.query_expander import QueryExpander
from retrieval.reranker import Reranker
from utils.config import (
    EMBEDDING_MODEL,
    ES_HOST,
    ES_INDEX,
    FUSION_TOP_N,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_TOP_K,
    RETRIEVAL_K,
    RRF_K,
)


class HybridRetriever:
    _encoder_cache = {}

    def __init__(self):
        self.collection = QDRANT_COLLECTION
        self.es_index = ES_INDEX
        self.k = RETRIEVAL_K
        self.rrf_k = RRF_K
        self.fusion_top_n = FUSION_TOP_N
        self.top_k = RERANK_TOP_K
        self.failure_threshold = 3
        self.recovery_interval_sec = 30

        self.qdrant = None
        self.es = None
        self.qdrant_healthy = False
        self.es_healthy = False
        self.qdrant_failure_count = 0
        self.es_failure_count = 0
        self.qdrant_last_failed_at = 0.0
        self.es_last_failed_at = 0.0

        self._init_qdrant()
        self._init_es()

        if EMBEDDING_MODEL not in self._encoder_cache:
            self._encoder_cache[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
        self.encoder = self._encoder_cache[EMBEDDING_MODEL]
        self.reranker = Reranker()
        self.query_expander = QueryExpander()

    def _init_qdrant(self):
        try:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.qdrant.get_collection(self.collection)
            self.qdrant_healthy = True
            self.qdrant_failure_count = 0
            self.qdrant_last_failed_at = 0.0
        except Exception:
            self.qdrant_healthy = False
            self.qdrant = None

    def _init_es(self):
        try:
            self.es = Elasticsearch(ES_HOST)
            if not self.es.ping():
                raise RuntimeError("Elasticsearch ping 失败")
            if not self.es.indices.exists(index=self.es_index):
                raise RuntimeError(f"Elasticsearch 索引不存在: {self.es_index}")
            self.es_healthy = True
            self.es_failure_count = 0
            self.es_last_failed_at = 0.0
        except Exception:
            self.es_healthy = False
            self.es = None

    def _mark_qdrant_failure(self):
        self.qdrant_failure_count += 1
        self.qdrant_last_failed_at = time.time()
        if self.qdrant_failure_count >= self.failure_threshold:
            self.qdrant_healthy = False

    def _mark_es_failure(self):
        self.es_failure_count += 1
        self.es_last_failed_at = time.time()
        if self.es_failure_count >= self.failure_threshold:
            self.es_healthy = False

    def _maybe_recover(self):
        now = time.time()
        if (not self.qdrant_healthy) and self.qdrant_last_failed_at and now - self.qdrant_last_failed_at >= self.recovery_interval_sec:
            self._init_qdrant()
        if (not self.es_healthy) and self.es_last_failed_at and now - self.es_last_failed_at >= self.recovery_interval_sec:
            self._init_es()

    def _vector_search(self, query, source_filter=None):
        if not self.qdrant_healthy or self.qdrant is None:
            return []

        try:
            q_vec = self.encoder.encode(query, normalize_embeddings=True).tolist()
            search_kwargs = {
                "collection_name": self.collection,
                "query_vector": q_vec,
                "limit": self.k,
            }
            if source_filter:
                search_kwargs["query_filter"] = Filter(
                    should=[
                        FieldCondition(key="source", match=MatchValue(value=src))
                        for src in source_filter
                    ]
                )
            hits = self.qdrant.search(**search_kwargs)
            docs = []
            for hit in hits:
                payload = hit.payload or {}
                chunk_id = payload.get("id", hit.id)
                text = payload.get("text", "")
                if text:
                    docs.append(
                        {
                            "id": str(chunk_id),
                            "text": text,
                            "chapter_id": payload.get("chapter_id", 0),
                            "chapter_title": payload.get("chapter_title", ""),
                        }
                    )
            self.qdrant_failure_count = 0
            return docs
        except Exception:
            self._mark_qdrant_failure()
            return []

    def _bm25_search(self, query, source_filter=None):
        if not self.es_healthy or self.es is None:
            return []

        try:
            query_jieba = " ".join(jieba.cut_for_search(query))
            if source_filter:
                query_body = {
                    "bool": {
                        "must": [{"match": {"text_jieba": query_jieba}}],
                        "filter": [{"terms": {"source": source_filter}}],
                    }
                }
            else:
                query_body = {"match": {"text_jieba": query_jieba}}

            resp = self.es.search(
                index=self.es_index,
                body={"query": query_body, "size": self.k},
            )
            docs = []
            for hit in resp["hits"]["hits"]:
                source = hit.get("_source", {})
                text = source.get("text", "")
                if text:
                    docs.append(
                        {
                            "id": str(hit.get("_id")),
                            "text": text,
                            "chapter_id": source.get("chapter_id", 0),
                            "chapter_title": source.get("chapter_title", ""),
                        }
                    )
            self.es_failure_count = 0
            return docs
        except Exception:
            self._mark_es_failure()
            return []

    def _fuse_and_rerank(self, query, vector_docs, bm25_docs):
        scores = {}
        id_to_doc = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = doc["id"]
            id_to_doc[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (self.rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc["id"]
            id_to_doc[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (self.rrf_k + rank + 1)

        candidate_ids = [
            doc_id
            for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.fusion_top_n]
        ]
        candidates = [id_to_doc[doc_id] for doc_id in candidate_ids]
        return self.reranker.rerank(query, candidates, top_k=self.top_k)

    def _retrieve_with_details(self, query, source_filter=None):
        self._maybe_recover()
        vector_docs = self._vector_search(query, source_filter=source_filter)
        bm25_docs = self._bm25_search(query, source_filter=source_filter)

        if not vector_docs and not bm25_docs:
            return []

        if vector_docs and bm25_docs:
            return self._fuse_and_rerank(query, vector_docs, bm25_docs)

        single_source_docs = vector_docs if vector_docs else bm25_docs
        reranked = self.reranker.rerank(query, single_source_docs, top_k=self.top_k)
        return reranked or single_source_docs[: self.top_k]

    def retrieve(self, query, source_filter=None):
        # 暂不启用 query expansion，先保持单查询检索链路稳定
        expanded_queries = [query]

        merged = {}
        for q in expanded_queries:
            docs = self._retrieve_with_details(q, source_filter=source_filter)
            for rank, doc in enumerate(docs):
                doc_id = doc["id"]
                score = 1 / (self.rrf_k + rank + 1)
                if doc_id not in merged or score > merged[doc_id]["score"]:
                    merged[doc_id] = {"doc": doc, "score": score}

        final_docs = [
            item["doc"]
            for item in sorted(merged.values(), key=lambda x: x["score"], reverse=True)[: self.top_k]
        ]
        return [doc["text"] for doc in final_docs]

    def retrieve_with_details(self, query, source_filter=None):
        # 暂不启用 query expansion，先保持单查询检索链路稳定
        expanded_queries = [query]

        merged = {}
        for q in expanded_queries:
            docs = self._retrieve_with_details(q, source_filter=source_filter)
            for rank, doc in enumerate(docs):
                doc_id = doc["id"]
                score = 1 / (self.rrf_k + rank + 1)
                if doc_id not in merged or score > merged[doc_id]["score"]:
                    merged[doc_id] = {"doc": doc, "score": score}

        final_docs = [
            item["doc"]
            for item in sorted(merged.values(), key=lambda x: x["score"], reverse=True)[: self.top_k]
        ]
        return final_docs
