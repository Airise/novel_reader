import json
import os
import re
import sys
import time
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 允许直接运行 test_agent.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever
from utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=DEEPSEEK_MODEL,
    temperature=0,
)
retriever = HybridRetriever()

MAX_CONTEXTS_FOR_GENERATION = 12
MAX_NO_GAIN_ROUNDS = 2
MAX_SUB_QUESTIONS = 3


def _load_prompt(path: str) -> PromptTemplate:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return PromptTemplate.from_template(text)


planner_prompt = _load_prompt("prompts/planner_prompt.txt")
reflector_prompt = _load_prompt("prompts/reflector_prompt.txt")


def _dedup_contexts(contexts: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for c in contexts:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _safe_parse_planner_action(raw_text: str) -> Dict:
    text = (raw_text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 兼容模型输出带解释 + JSON 的情况
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {"action": "refine"}


def _normalize_query(query: str) -> str:
    """归一化 query，用于去重（大小写、空白、常见标点）。"""
    q = (query or "").strip().lower()
    q = re.sub(r"\s+", "", q)
    q = re.sub(r"[，。！？；：、“”‘’（）()【】\[\]{}<>《》,.;:!?\-_/]", "", q)
    return q


def _compress_query(query: str, max_terms: int = 10, max_chars: int = 48) -> str:
    """压缩 query，避免越搜越长导致召回漂移。"""
    q = (query or "").strip()
    if not q:
        return q

    terms = [t for t in re.split(r"\s+", q) if t]
    if len(terms) > 1:
        dedup_terms = []
        seen = set()
        for t in terms:
            nt = _normalize_query(t)
            if nt and nt not in seen:
                dedup_terms.append(t)
                seen.add(nt)
        return " ".join(dedup_terms[:max_terms])[:max_chars].strip()

    return q[:max_chars].strip()


def _extract_core_terms(question: str, max_terms: int = 4) -> List[str]:
    """从问题中提取核心词（不依赖特定题型）。"""
    text = re.sub(r"[，。！？；：、“”‘’（）()【】\[\]{}<>《》,.;:!?\-_/]", " ", question)
    parts = [p.strip() for p in re.split(r"\s+", text) if p.strip()]

    # 优先保留更长的词，减少停用碎词干扰
    parts = sorted(parts, key=lambda x: len(x), reverse=True)
    result = []
    seen = set()
    for p in parts:
        n = _normalize_query(p)
        if n and n not in seen:
            result.append(p)
            seen.add(n)
        if len(result) >= max_terms:
            break
    return result


def _build_bridging_query(question: str, tag: str) -> str:
    """根据连续失败标签构造桥接式 query（泛用）。"""
    cores = _extract_core_terms(question)
    core_text = " ".join(cores) if cores else question

    bridge_map = {
        "missing_entity": "实体 身份 别名 指代 对应",
        "missing_relation": "关系 关联 因果 前因后果 直接原因",
        "missing_time": "时间 顺序 阶段 前后 发生时间",
        "unclear": "定义 关键事实 核心证据",
    }
    bridge = bridge_map.get(tag, bridge_map["unclear"])
    return _compress_query(f"{core_text} {bridge}")


def _classify_query_type(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "fact"

    contradiction_markers = ["是否矛盾", "矛盾", "冲突", "不一致", "前后是否一致", "前后是否冲突"]
    comparison_markers = ["对比", "比较", "区别", "不同", "一致", "是否一致", "有哪些异同"]
    tracing_markers = ["后来", "之后", "结局", "最终", "最后", "下场", "结果", "发展变化", "前后变化", "演变"]
    causal_markers = ["为什么", "原因", "如何", "怎么", "怎样", "过程", "步骤", "策略", "手段", "办法", "计划", "通过什么", "如何实现"]
    search_markers = ["找出", "列出", "所有", "哪些", "哪里", "在哪", "出处", "原文", "全文检索"]

    if any(k in q for k in contradiction_markers):
        return "contradiction"
    if any(k in q for k in comparison_markers):
        return "comparison"
    if any(k in q for k in tracing_markers):
        return "tracing"
    if any(k in q for k in causal_markers):
        return "causal"
    if any(k in q for k in search_markers):
        return "search"
    return "fact"


def _generate_sub_questions(question: str, query_type: str) -> List[str]:
    q = (question or "").strip()
    if query_type == "causal":
        subs = [
            f"{q} 的关键步骤",
            f"{q} 依赖的关键人物或证据",
            f"{q} 的结果和影响",
        ]
    elif query_type == "tracing":
        subs = [
            f"{q} 的关键时间线",
            f"{q} 中涉及的主要人物关系变化",
            f"{q} 的最终结果或结局",
        ]
    elif query_type == "contradiction":
        subs = [
            f"{q} 涉及的前后两处原文证据",
            f"{q} 中是否存在人物、时间或设定冲突",
            f"{q} 对应的章节或段落差异",
        ]
    elif query_type == "comparison":
        subs = [
            f"{q} 的两个对象各自描述",
            f"{q} 的共同点与不同点",
            f"{q} 的结论依据",
        ]
    else:
        subs = [q]

    cleaned = []
    seen = set()
    for item in subs:
        item = _compress_query(item)
        norm = _normalize_query(item)
        if item and norm not in seen:
            cleaned.append(item)
            seen.add(norm)
        if len(cleaned) >= MAX_SUB_QUESTIONS:
            break
    return cleaned


def router_node(state: AgentState) -> AgentState:
    state["query_type"] = _classify_query_type(state["question"])
    state["sub_questions"] = _generate_sub_questions(state["question"], state["query_type"])
    state["degraded_mode"] = False
    return state


def planner_node(state: AgentState) -> AgentState:
    """规划下一步：search / refine"""
    t0 = time.perf_counter()
    contexts_summary = "\n\n".join(state["contexts"][-5:]) if state["contexts"] else "暂无"
    recent_steps = state["steps"][-6:] if state["steps"] else []
    steps_str = "\n".join(recent_steps) if recent_steps else "暂无"
    source_filter = state.get("source_filter") or []

    verify_feedback = (state.get("verifier_feedback") or "").strip()
    feedback_tag = (state.get("verifier_feedback_tag") or "").strip()
    planner_extra = ""
    if verify_feedback:
        planner_extra += f"\n\n上轮校验反馈（用于改进下一轮检索）：\n{verify_feedback}"
    if feedback_tag:
        planner_extra += f"\n校验缺陷标签：{feedback_tag}"
    if source_filter:
        planner_extra += f"\n当前来源过滤：{', '.join(source_filter)}"

    chain = planner_prompt | llm
    response = chain.invoke(
        {
            "question": state["question"],
            "contexts": contexts_summary,
            "steps": steps_str + planner_extra,
        }
    )
    state["planner_raw"] = response.content or ""

    action = _safe_parse_planner_action(response.content)
    act = (action.get("action") or "").strip().lower()

    if act == "search":
        query = (action.get("query") or "").strip()
        if not query:
            action = {"action": "refine"}
        else:
            action = {"action": "search", "query": query}
    else:
        action = {"action": "refine"}

    # 若上一轮校验 FAIL，但本轮仍给 refine，强制回退为 search（避免 planner-reflector 空转）
    last_verify = (state.get("verifier_raw") or "").upper()
    if action.get("action") == "refine" and "FAIL" in last_verify:
        fallback_query = state["question"].strip()
        action = {"action": "search", "query": fallback_query}
        state["steps"].append("规划: 上轮校验失败，回退为补充检索")

    # 第三个泛用补丁：连续两轮同标签 FAIL 时，强制桥接式 query（策略切换）
    tag_history = state.get("verifier_tag_history") or []
    if len(tag_history) >= 2 and tag_history[-1] == tag_history[-2] and tag_history[-1] != "unclear":
        forced_query = _build_bridging_query(state["question"], tag_history[-1])
        action = {"action": "search", "query": forced_query}
        state["steps"].append(
            f"规划: 连续两轮同类缺陷({tag_history[-1]})，启用桥接检索"
        )

    # 重复 query 检测：基于归一化 query 去重，避免“同义重写但语义不变”空转
    if action.get("action") == "search":
        query = _compress_query(action["query"])
        action["query"] = query
        normalized = _normalize_query(query)
        normalized_history = {_normalize_query(q) for q in state["searched_queries"]}
        if normalized in normalized_history:
            state["steps"].append(f"规划: 检测到重复查询（归一化后），跳过（{query}）")
            action = {"action": "refine"}
            state["no_gain_rounds"] = state.get("no_gain_rounds", 0) + 1
        else:
            state["searched_queries"].append(query)

    state["current_action"] = action
    state["node_timings_ms"]["planner"].append(round((time.perf_counter() - t0) * 1000, 2))
    return state


def executor_node(state: AgentState) -> AgentState:
    """执行检索并更新状态"""
    t0 = time.perf_counter()
    action: Dict = state.get("current_action") or {}

    queries: List[str] = []
    if action.get("action") == "search" and (action.get("query") or "").strip():
        queries.append((action.get("query") or "").strip())

    query_type = state.get("query_type")
    if query_type in {"tracing", "comparison", "contradiction", "causal"}:
        for sub_q in state.get("sub_questions") or []:
            sub_q = (sub_q or "").strip()
            if sub_q:
                queries.append(sub_q)

    queries = _dedup_contexts(queries)
    if not queries and state.get("question"):
        queries = [state["question"].strip()]

    if not queries:
        state["steps"].append("搜索: 空查询，跳过")
        state["node_timings_ms"]["executor"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    source_filter = state.get("source_filter") or []
    retrieval_items = []
    collected_chapters = []

    for q in queries[:MAX_SUB_QUESTIONS]:
        try:
            docs = retriever.retrieve_with_details(q, source_filter=source_filter)
            new_contexts = [doc.get("text", "") for doc in docs if doc.get("text")]
        except Exception as e:
            msg = f"搜索失败: {q} ({e})"
            state["steps"].append(msg)
            state["errors"].append(msg)
            retrieval_items.append({"query": q, "retrieved": 0, "added": 0, "error": str(e), "query_type": state.get("query_type")})
            state["no_gain_rounds"] += 1
            continue

        before = len(state["contexts"])
        state["contexts"] = _dedup_contexts(state["contexts"] + new_contexts)
        added = len(state["contexts"]) - before

        retrieval_items.append(
            {
                "query": q,
                "source_filter": source_filter,
                "retrieved": len(new_contexts),
                "added": added,
                "error": None,
                "query_type": state.get("query_type"),
            }
        )

        for doc in docs:
            chap = doc.get("chapter_id")
            if chap is not None:
                collected_chapters.append(str(chap))

        if added == 0:
            state["no_gain_rounds"] += 1
        else:
            state["no_gain_rounds"] = 0

        state["steps"].append(f"搜索: {q} (新增片段 {added})")
        state["step_count"] += 1

    state["retrieval_trace"].extend(retrieval_items)
    state["sub_question_results"] = retrieval_items
    state["degraded_mode"] = bool((not getattr(retriever, "qdrant_healthy", True)) or (not getattr(retriever, "es_healthy", True)))
    state["chapter_candidates"] = _dedup_contexts(collected_chapters)

    state["node_timings_ms"]["executor"].append(round((time.perf_counter() - t0) * 1000, 2))
    return state


def reflector_node(state: AgentState) -> AgentState:
    """反思是否继续检索"""
    t0 = time.perf_counter()
    if state["step_count"] >= state["max_steps"]:
        state["should_continue"] = False
        state["steps"].append("达到最大步数，停止检索")
        state["node_timings_ms"]["reflector"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    max_no_gain_rounds = state.get("max_no_gain_rounds", MAX_NO_GAIN_ROUNDS)
    if state["no_gain_rounds"] >= max_no_gain_rounds:
        state["should_continue"] = False
        state["steps"].append("反思: 连续低增益检索，提前停止")
        state["node_timings_ms"]["reflector"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    contexts_str = "\n\n".join(state["contexts"][-8:]) if state["contexts"] else "暂无"

    chain = reflector_prompt | llm
    response = chain.invoke({"question": state["question"], "contexts": contexts_str})
    state["reflector_raw"] = response.content or ""
    decision = (response.content or "").strip().upper()

    # 宽松解析：只要出现 NO 就继续，出现 YES 就停止；否则默认继续一轮
    if "NO" in decision:
        state["should_continue"] = True
        state["steps"].append("反思: 信息不足，继续检索")
    elif "YES" in decision:
        state["should_continue"] = False
        state["steps"].append("反思: 信息充分，进入生成")
    else:
        state["should_continue"] = True
        state["steps"].append(f"反思: 输出不规范（{decision}），默认继续")

    # 若上轮校验 FAIL，但反思仍判断 YES，则强制继续一轮（快速模式关闭此回环）
    last_verify = (state.get("verifier_raw") or "").upper()
    if (not state.get("fast_mode", False)) and "FAIL" in last_verify and state["should_continue"] is False:
        state["should_continue"] = True
        state["steps"].append("反思: 上轮校验失败，强制继续检索")

    state["node_timings_ms"]["reflector"].append(round((time.perf_counter() - t0) * 1000, 2))
    return state


def generator_node(state: AgentState) -> AgentState:
    """最终答案生成（按问题类型自适应输出）。"""
    t0 = time.perf_counter()
    if not state["contexts"]:
        state["answer"] = "未找到相关信息。"
        state["node_timings_ms"]["generator"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    selected_contexts = state["contexts"][-MAX_CONTEXTS_FOR_GENERATION:]
    contexts_str = "\n\n".join(selected_contexts)
    question = state["question"]
    query_type = state.get("query_type", "fact")
    sub_questions = state.get("sub_questions") or []
    degraded_mode = state.get("degraded_mode", False)

    if query_type == "tracing":
        format_hint = """请按以下格式输出：
1) 时间线（按先后顺序列出 2-5 条）
2) 关键人物变化
3) 最终结局/状态
4) 结论（若证据不足，请明确说明）"""
    elif query_type == "contradiction":
        format_hint = """请按以下格式输出：
1) 证据A
2) 证据B
3) 是否矛盾（是/否/无法判断）
4) 矛盾点说明
5) 结论"""
    elif query_type == "comparison":
        format_hint = """请按以下格式输出：
1) 对象A
2) 对象B
3) 相同点
4) 不同点
5) 结论"""
    else:
        format_hint = """请按“先事实、后推理、再结论”的方式作答，并严格输出以下结构：
1) 关键事实（1-4条）：每条必须是可直接从信息中抽取的事实，包含实体与关系/事件，并在末尾附上原文短引（8-20字）。
2) 推理链：说明你如何由关键事实推出结论（若无法推出，明确缺少哪条事实）。
3) 结论类型：从【确定 / 倾向 / 信息不足】中选一项。
4) 简洁答案：一句话。
5) 关键依据：列出支撑结论的 1-3 条证据（必须与上文短引一致）。
6) 若证据冲突，请单独指出冲突点。"""

    subq_text = "\n".join(f"- {q}" for q in sub_questions) if sub_questions else "暂无"
    degraded_hint = "当前处于检索降级模式，请更谨慎作答并明确标注不确定性。" if degraded_mode else ""

    prompt = f"""你是一个严谨的中文小说问答与审校助手。
请只基于给定信息回答问题；若信息不足，明确回答“我不知道”。
{degraded_hint}

问题：{question}
问题类型：{query_type}
子问题：
{subq_text}

信息：
{contexts_str}

{format_hint}

硬性要求：
- 不得引入信息中未出现的人物、事件或设定。
- 若找不到可引用的原文短引，必须输出“信息不足”。
- 如果子问题检索结果之间存在冲突，请明确指出冲突来源。"""

    response = llm.invoke(prompt)
    state["answer"] = (response.content or "").strip()
    state["node_timings_ms"]["generator"].append(round((time.perf_counter() - t0) * 1000, 2))
    return state


def _classify_feedback_tag(feedback: str) -> str:
    f = (feedback or "").strip()
    if not f:
        return "unclear"

    rules = {
        "missing_entity": ["实体", "人物", "角色", "名称", "别名", "身份"],
        "missing_relation": ["关系", "关联", "因果", "桥梁", "链", "亲缘"],
        "missing_time": ["时间", "先后", "时序", "阶段", "年", "月", "日"],
    }

    for tag, kws in rules.items():
        if any(kw in f for kw in kws):
            return tag
    return "unclear"


def verifier_node(state: AgentState) -> AgentState:
    """对生成答案进行一致性校验，不通过则继续检索。"""
    t0 = time.perf_counter()

    if state.get("fast_mode", False):
        state["verify_pass"] = True
        state["verifier_raw"] = "SKIPPED_IN_FAST_MODE"
        state["verifier_feedback"] = "快速模式已跳过校验"
        state["verifier_feedback_tag"] = "unclear"
        state["node_timings_ms"]["verifier"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    answer = (state.get("answer") or "").strip()
    contexts = state.get("contexts") or []
    question = state.get("question") or ""
    query_type = state.get("query_type", "fact")

    if not answer:
        state["verify_pass"] = False
        state["verifier_raw"] = "EMPTY_ANSWER"
        state["verifier_feedback"] = "答案为空"
        state["verifier_feedback_tag"] = "unclear"
        state["should_continue"] = True
        state["steps"].append("校验: 空答案，返回继续检索")
        state["node_timings_ms"]["verifier"].append(round((time.perf_counter() - t0) * 1000, 2))
        return state

    contexts_str = "\n\n".join(contexts[-10:]) if contexts else "暂无"

    if query_type == "tracing":
        rule_hint = "重点检查时间线是否前后连贯、是否遗漏关键节点。"
    elif query_type == "contradiction":
        rule_hint = "重点检查答案是否准确指出证据冲突，避免把无冲突判成冲突。"
    elif query_type == "comparison":
        rule_hint = "重点检查对比项是否完整、是否混淆比较对象。"
    else:
        rule_hint = "重点检查事实一致性、推理完整性与结论审慎性。"

    prompt = f"""你是严格的答案校验器。请判断答案是否被给定信息支持。

问题：{question}
问题类型：{query_type}

候选答案：
{answer}

证据信息：
{contexts_str}

校验重点：
{rule_hint}

判定规则（通用）：
1) 事实一致性：答案中的关键事实是否能在证据中找到支撑。
2) 推理完整性：答案是否说明了从事实到结论的关键推理步骤。
3) 结论审慎性：若证据不足，答案是否明确表达不确定，而非武断下结论。

输出格式（严格）：
- 第一行输出 PASS 或 FAIL
- 第二行输出一句原因（不超过30字）
- 第三行输出缺陷标签：missing_entity / missing_relation / missing_time / unclear

额外校验（严格）：
- 若答案出现了证据中不存在的人名/事件词（疑似幻觉），必须判定 FAIL，原因写“出现证据外信息”，标签用 missing_entity。

示例：
PASS
证据充分且推理链完整
unclear

或
FAIL
缺少关键关系桥梁事实
missing_relation
"""

    response = llm.invoke(prompt)
    raw = (response.content or "").strip()
    state["verifier_raw"] = raw

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    verdict = lines[0].upper() if lines else "FAIL"
    feedback = lines[1] if len(lines) >= 2 else "缺少明确校验原因"

    tag_line = lines[2].lower() if len(lines) >= 3 else ""
    valid_tags = {"missing_entity", "missing_relation", "missing_time", "unclear"}
    tag = tag_line if tag_line in valid_tags else _classify_feedback_tag(feedback)

    state["verifier_feedback"] = feedback
    state["verifier_feedback_tag"] = tag
    state["verifier_tag_history"].append(tag)

    if verdict.startswith("PASS"):
        state["verify_pass"] = True
        state["should_continue"] = False
        state["steps"].append("校验: PASS，结束")
    else:
        state["verify_pass"] = False
        state["should_continue"] = state["step_count"] < state["max_steps"]
        state["steps"].append(f"校验: FAIL，返回继续检索（原因: {feedback}，标签: {tag}）")

    state["node_timings_ms"]["verifier"].append(round((time.perf_counter() - t0) * 1000, 2))
    return state