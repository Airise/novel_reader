from openai import OpenAI

from retrieval.hybrid_retriever import HybridRetriever
from utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class SingleRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.last_contexts = []
        try:
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        except Exception as e:
            raise RuntimeError(f"DeepSeek 客户端初始化失败: {e}") from e

    def answer(self, question):
        try:
            contexts = self.retriever.retrieve(question)
            self.last_contexts = contexts
        except Exception as e:
            self.last_contexts = []
            return f"检索失败：{e}"

        if not contexts:
            return "未找到相关信息。"

        context_str = "\n\n".join(contexts)
        prompt = f"""根据以下信息回答问题。如果信息不足，回答"不知道"。

信息：
{context_str}

问题：{question}

答案："""

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个乐于助人的助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"生成失败：DeepSeek 服务不可用或请求异常（{e}）"