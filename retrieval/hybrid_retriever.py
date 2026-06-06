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

        try:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.qdrant.get_collection(self.collection)
        except Exception as e:
            raise RuntimeError(
                f"Qdrant 不可用或集合不存在: {self.collection}，错误: {e}"
            ) from e

        try:
            self.es = Elasticsearch(ES_HOST)
            if not self.es.ping():
                raise RuntimeError("Elasticsearch ping 失败")
            if not self.es.indices.exists(index=self.es_index):
                raise RuntimeError(f"Elasticsearch 索引不存在: {self.es_index}")
        except Exception as e:
            raise RuntimeError(f"Elasticsearch 不可用，错误: {e}") from e

        if EMBEDDING_MODEL not in self._encoder_cache:
            self._encoder_cache[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
        self.encoder = self._encoder_cache[EMBEDDING_MODEL]
        self.reranker = Reranker()
        self.query_expander = QueryExpander()

    def _vector_search(self, query, source_filter=None):
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
                    docs.append({"id": str(chunk_id), "text": text})
            return docs
        except Exception as e:
            raise RuntimeError(f"Qdrant 检索失败，错误: {e}") from e

    def _bm25_search(self, query, source_filter=None):
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
                    docs.append({"id": str(hit.get("_id")), "text": text})
            return docs
        except Exception as e:
            raise RuntimeError(f"Elasticsearch 检索失败，错误: {e}") from e

    def _retrieve_with_details(self, query, source_filter=None):
        vector_docs = self._vector_search(query, source_filter=source_filter)
        bm25_docs = self._bm25_search(query, source_filter=source_filter)

        # RRF 融合（按稳定 chunk_id 融合）
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
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        return reranked

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