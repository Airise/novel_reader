from sentence_transformers import CrossEncoder
from utils.config import RERANKER_MODEL, RERANK_MIN_SCORE


class Reranker:
    _model_cache = {}

    def __init__(self, model_name=RERANKER_MODEL):
        self.model_name = model_name
        if model_name not in self._model_cache:
            self._model_cache[model_name] = CrossEncoder(model_name)
        self.model = self._model_cache[model_name]

    def rerank(self, query, documents, top_k=3, min_score=RERANK_MIN_SCORE):
        if not documents:
            return []
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        filtered = [(doc, score) for doc, score in ranked if score >= min_score]
        if not filtered:
            # 若全部低于阈值，保底返回 top1，避免完全无上下文
            return [ranked[0][0]] if ranked else []

        return [doc for doc, _ in filtered[:top_k]]