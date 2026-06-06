import re
from typing import List, Set


class QueryExpander:
    """增强版中文查询扩展器。

    目标：在不依赖额外在线 LLM 的前提下，提供可控、可解释的多查询改写。
    """

    def __init__(self, max_queries: int = 12):
        self.max_queries = max_queries

    # 领域同义词（可按小说继续补）
    SYNONYM_MAP = {
        "扳倒": ["斗倒", "拉下马", "扳垮", "弹劾", "构陷", "定罪", "翻案"],
        "怎么": ["如何", "经过", "过程", "具体是怎样"],
        "为什么": ["原因", "缘由", "动机", "为何"],
        "真相": ["内情", "原委", "来龙去脉", "内幕"],
        "证据": ["线索", "凭据", "把柄", "证词"],
        "陷害": ["构陷", "栽赃", "嫁祸"],
        "计划": ["谋划", "布局", "安排", "筹谋"],
        "帮助": ["协助", "相助", "支援"],
        "关系": ["关联", "牵连", "联系"],
        "身份": ["来历", "背景", "身世"],
    }

    QUESTION_PREFIX = ["", "请概括", "请说明", "请详细说明"]
    QUESTION_SUFFIX = ["", "的关键过程", "的时间线", "涉及哪些人物与证据"]

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _safe_add(self, bucket: List[str], seen: Set[str], candidate: str):
        candidate = self._normalize(candidate)
        if not candidate:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        bucket.append(candidate)

    def _synonym_rewrite(self, query: str) -> List[str]:
        rewrites = []
        for key, synonyms in self.SYNONYM_MAP.items():
            if key in query:
                for syn in synonyms:
                    rewrites.append(query.replace(key, syn))
        return rewrites

    def _intent_templates(self, query: str) -> List[str]:
        """针对“过程型/原因型”问题生成模板查询。"""
        templates = []

        # 通用模板
        templates.extend(
            [
                f"{query} 关键情节",
                f"{query} 相关人物",
                f"{query} 前因后果",
                f"{query} 证据 线索",
            ]
        )

        # 疑问词模板
        if any(w in query for w in ["怎么", "如何", "经过", "过程"]):
            templates.extend(
                [
                    query.replace("怎么", "如何") if "怎么" in query else f"{query} 如何发生",
                    f"{query} 时间线",
                    f"{query} 具体步骤",
                ]
            )
        if any(w in query for w in ["为什么", "为何", "原因", "动机"]):
            templates.extend([f"{query} 动机", f"{query} 原因", f"{query} 背后目的"])

        return templates

    def _prefix_suffix_rewrite(self, query: str) -> List[str]:
        rewrites = []
        for p in self.QUESTION_PREFIX:
            for s in self.QUESTION_SUFFIX:
                if not p and not s:
                    continue
                rewrites.append(f"{p}{query}{s}")
        return rewrites

    def expand(self, query: str) -> List[str]:
        query = self._normalize(query)
        if not query:
            return []

        ordered_queries: List[str] = []
        seen: Set[str] = set()

        # 1) 原始问题优先
        self._safe_add(ordered_queries, seen, query)

        # 2) 同义词改写
        for q in self._synonym_rewrite(query):
            self._safe_add(ordered_queries, seen, q)

        # 3) 意图模板扩展
        for q in self._intent_templates(query):
            self._safe_add(ordered_queries, seen, q)

        # 4) 前后缀风格改写
        for q in self._prefix_suffix_rewrite(query):
            self._safe_add(ordered_queries, seen, q)

        return ordered_queries[: self.max_queries]
