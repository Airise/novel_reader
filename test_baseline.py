from openai import OpenAI
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

from baseline.single_rag import SingleRAG
from utils.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    ES_HOST,
    ES_INDEX,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)


def print_retrieved_contexts(contexts, max_show=5, max_chars=180):
    print("\n--- 检索到的 Top 上下文 ---")
    if not contexts:
        print("(空)")
        return
    for i, ctx in enumerate(contexts[:max_show], start=1):
        preview = ctx[:max_chars].replace("\n", " ")
        print(f"[{i}] {preview}")
    print("-------------------------")


def health_check() -> bool:
    print("\n================ 启动前健康检查 ================")
    ok = True

    # Qdrant
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant.get_collection(QDRANT_COLLECTION)
        print(f"[OK] Qdrant 可用，集合存在: {QDRANT_COLLECTION}")
    except Exception as e:
        ok = False
        print(f"[FAIL] Qdrant 异常: {e}")

    # Elasticsearch
    try:
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            raise RuntimeError("ping 失败")
        if not es.indices.exists(index=ES_INDEX):
            raise RuntimeError(f"索引不存在: {ES_INDEX}")
        print(f"[OK] Elasticsearch 可用，索引存在: {ES_INDEX}")
    except Exception as e:
        ok = False
        print(f"[FAIL] Elasticsearch 异常: {e}")

    # DeepSeek
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=5,
            temperature=0,
        )
        print(f"[OK] DeepSeek API 可用，模型: {DEEPSEEK_MODEL}")
    except Exception as e:
        ok = False
        print(f"[FAIL] DeepSeek API 异常: {e}")

    print("================================================")
    return ok


def test_baseline():
    if not health_check():
        print("\n健康检查未通过，请先修复上述问题再运行 baseline。")
        return

    rag = SingleRAG()
    while True:
        question = input("\n请输入你的问题（输入 q 退出）：")
        if question.lower() == "q":
            break
        answer = rag.answer(question)
        print_retrieved_contexts(rag.last_contexts, max_show=5, max_chars=180)
        print("答案：", answer)


if __name__ == "__main__":
    test_baseline()