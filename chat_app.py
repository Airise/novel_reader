import csv
import hashlib
import io
import json
import os
import re
import time
from typing import Any, Callable, List, Optional

import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from indexing.build_bm25_index import build_bm25_index
from indexing.build_vector_store import build_vector_store
from test_agent import run_agent
from utils.config import INDEX_DIR


st.set_page_config(page_title="Week2 Agent Chat", page_icon="🤖", layout="wide")


@st.cache_resource
def get_agent_runner():
    return run_agent


def init_state():
    defaults = {
        "messages": [],
        "processing": False,
        "processed_docs": [],
        "processed_summary": "未处理任何文件",
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    return text.strip()


def _decode_bytes_with_fallback(data: bytes) -> str:
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "gbk", "big5", "cp950", "cp936", "latin1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def _flatten_json_strings(obj: Any, out: List[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.append(k)
            _flatten_json_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _flatten_json_strings(item, out)
    elif isinstance(obj, (str, int, float, bool)):
        out.append(str(obj))


def read_uploaded_file(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith((".txt", ".md")):
        return _decode_bytes_with_fallback(data)
    if name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(data))
        return "\n\n".join([page.extract_text() or "" for page in pdf.pages])
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)
    if name.endswith(".csv"):
        text = _decode_bytes_with_fallback(data)
        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join([c.strip() for c in row if c is not None]) for row in reader]
        return "\n".join([r for r in rows if r.strip()])
    if name.endswith(".json"):
        text = _decode_bytes_with_fallback(data)
        try:
            obj = json.loads(text)
            out: List[str] = []
            _flatten_json_strings(obj, out)
            return "\n".join([s for s in out if s.strip()])
        except Exception:
            return text
    if name.endswith((".html", ".htm")):
        soup = BeautifulSoup(_decode_bytes_with_fallback(data), "html.parser")
        return soup.get_text("\n")
    return _decode_bytes_with_fallback(data)


def _chunk_unique_id(source: str, text: str) -> str:
    return hashlib.md5(f"{source}::{text}".encode("utf-8")).hexdigest()


def _normalize_existing_chunks(raw_chunks: List[dict]) -> List[dict]:
    normalized = []
    for item in raw_chunks:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        source = (item.get("source") or "unknown").strip() or "unknown"
        uid = item.get("uid") or _chunk_unique_id(source, text)
        normalized.append({"uid": uid, "text": text, "source": source})
    return normalized



def _split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])


def _chunk_paragraph(paragraph: str, max_chars: int, overlap: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    chunks: List[str] = []
    step = max(1, max_chars - overlap)
    start = 0
    while start < len(paragraph):
        c = paragraph[start : start + max_chars].strip()
        if c:
            chunks.append(c)
        if start + max_chars >= len(paragraph):
            break
        start += step
    return chunks


def save_uploaded_chunks_to_index(files, mode: str = "incremental", on_progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = os.path.join(INDEX_DIR, "novel_chunks.json")
    chapter_hashes_path = os.path.join(INDEX_DIR, "chapter_hashes.json")

    existing_chunks = []
    if mode == "incremental" and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fr:
            existing_chunks = _normalize_existing_chunks(json.load(fr))

    existing_ids = {item.get("uid") for item in existing_chunks if item.get("uid")}
    docs = []
    new_chunks = []
    chapter_hashes = []

    total_files = len(files)
    for i, f in enumerate(files, start=1):
        if on_progress:
            on_progress(i - 1, total_files, f"正在解析与切分：{f.name}")

        raw = read_uploaded_file(f)
        chapters = [
            {"title": "全文", "text": raw}
        ]
        added_for_doc = 0
        doc_chunk_count = 0

        for ch_idx, ch in enumerate(chapters, start=1):
            chapter_title = ch["title"]
            chapter_text = normalize_text(ch["text"])
            chapter_hash = hashlib.md5(chapter_text.encode("utf-8")).hexdigest()
            chapter_hashes.append({"source": f.name, "chapter_title": chapter_title, "chapter_hash": chapter_hash})

            paras = _split_paragraphs(chapter_text)
            local_chunks: List[str] = []
            for p in paras:
                local_chunks.extend(_chunk_paragraph(p, max_chars=320, overlap=80))

            for c in local_chunks:
                doc_chunk_count += 1
                uid = _chunk_unique_id(f.name, c)
                if mode == "incremental" and uid in existing_ids:
                    continue
                new_chunks.append({"uid": uid, "text": c, "source": f.name, "chapter_title": chapter_title, "chapter_hash": chapter_hash, "chapter_id": ch_idx})
                existing_ids.add(uid)
                added_for_doc += 1

        docs.append({"name": f.name, "size": f.size, "chunk_count": doc_chunk_count, "added_chunk_count": added_for_doc, "chapter_count": len(chapters)})
        if on_progress:
            on_progress(i, total_files, f"已完成解析：{f.name}（{i}/{total_files}）")

    final_chunks = new_chunks if mode == "overwrite" else existing_chunks + new_chunks
    chunk_data = []
    for idx, item in enumerate(final_chunks):
        chunk_data.append({"id": idx, "uid": item["uid"], "text": item["text"], "source": item["source"], "chapter_id": item.get("chapter_id", 0), "chapter_title": item.get("chapter_title", ""), "chapter_hash": item.get("chapter_hash", ""), "chunk_hash": hashlib.md5(item["text"].encode("utf-8")).hexdigest()})

    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(chunk_data, fw, ensure_ascii=False, indent=2)
    with open(chapter_hashes_path, "w", encoding="utf-8") as fw:
        json.dump(chapter_hashes, fw, ensure_ascii=False, indent=2)

    build_vector_store()
    build_bm25_index()

    return {"docs": docs, "new_chunks": len(new_chunks), "total_chunks": len(chunk_data), "mode": mode, "output_path": out_path}


def get_available_sources() -> List[str]:
    path = os.path.join(INDEX_DIR, "novel_chunks.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fr:
            chunks = json.load(fr)
    except Exception:
        return []
    sources = sorted({(item.get("source") or "").strip() for item in chunks if item.get("source")})
    return [s for s in sources if s]


st.title("NovelQuest Agentic RAG 对话")
st.caption("已接入 NovelQuest Agentic 后端；支持文件上传并手动触发处理")

with st.sidebar:
    st.subheader("运行设置")
    fast_mode = st.toggle("快速模式（max_steps=3，跳过 verifier 二次回环）", value=True)
    if fast_mode:
        max_steps = 3
        st.caption("快速模式已启用：max_steps 固定为 3")
    else:
        max_steps = st.slider("max_steps", min_value=2, max_value=10, value=6, step=1)

    show_debug = st.toggle("显示调试信息（steps/trace）", value=True)
    st.divider()
    st.subheader("文件上传与处理")

    index_mode = st.radio("索引模式", options=["incremental", "overwrite"], format_func=lambda x: "增量追加" if x == "incremental" else "覆盖重建", horizontal=True)
    files = st.file_uploader("上传文档（支持 txt/md/pdf/docx/csv/json/html）", type=["txt", "md", "pdf", "docx", "csv", "json", "html", "htm"], accept_multiple_files=True, disabled=st.session_state.processing)
    process_btn = st.button("开始处理文件", use_container_width=True, disabled=st.session_state.processing or not files)

    if process_btn and files:
        st.session_state.processing = True
        progress = st.progress(0, text="准备开始处理...")
        try:
            parse_weight = 70

            def _on_parse_progress(done: int, total: int, message: str) -> None:
                ratio = (done / total) if total else 0
                progress.progress(int(parse_weight * ratio), text=message)

            result = save_uploaded_chunks_to_index(files, mode=index_mode, on_progress=_on_parse_progress)
            progress.progress(85, text="正在重建向量索引（Qdrant）...")
            progress.progress(95, text="正在重建 BM25 索引（Elasticsearch）...")
            progress.progress(100, text="处理完成")

            st.session_state.processed_docs = result["docs"]
            st.session_state.processed_summary = f"模式: {'增量追加' if result['mode'] == 'incremental' else '覆盖重建'}；本次新增 {result['new_chunks']} 个文本块，当前总计 {result['total_chunks']} 个；已写入 {result['output_path']} 并重建索引"
            st.success(st.session_state.processed_summary)
        except Exception as e:
            st.error(f"处理失败：{e}")
        finally:
            st.session_state.processing = False

    if st.session_state.processing:
        st.warning("文件处理中：暂时不能删除/修改上传文件")
    st.info(st.session_state.processed_summary)

    if st.session_state.processed_docs:
        with st.expander("查看处理结果", expanded=False):
            for d in st.session_state.processed_docs:
                st.write(f"- {d['name']} | {d['chapter_count']} chapters | {d['chunk_count']} chunks | {d['size']} bytes")

    source_options = get_available_sources()
    selected_sources = st.multiselect("提问来源过滤（可选，多选）", options=source_options, default=[], disabled=st.session_state.processing, help="自动读取已入库 source 列表；不选表示全库检索")
    if st.button("清空对话", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


st.sidebar.markdown("---")
st.sidebar.subheader("对话提示")
st.sidebar.caption("建议优先测试：如何/为什么/前后是否一致/是否矛盾/最后结局 这类问题。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("请输入你的问题...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在多步检索与推理..."):
            t0 = time.perf_counter()
            agent_runner = get_agent_runner()
            source_filter = selected_sources or []
            result = agent_runner(user_input, max_steps=max_steps, fast_mode=fast_mode, source_filter=source_filter)
            st.session_state.last_result = result
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

        answer = result.get("answer") or "（无回答）"
        st.markdown(answer)

        meta_cols = st.columns(4)
        meta_cols[0].metric("问题类型", result.get("query_type", "fact"))
        meta_cols[1].metric("是否降级", "是" if result.get("degraded_mode") else "否")
        meta_cols[2].metric("子问题数", str(len(result.get("sub_questions") or [])))
        meta_cols[3].metric("总耗时", f"{total_ms} ms")

        if result.get("sub_questions"):
            with st.expander("查看子问题", expanded=False):
                for q in result.get("sub_questions", []):
                    st.write(f"- {q}")

        timings = result.get("node_timings_ms", {})
        timing_cols = st.columns(5)
        for i, node in enumerate(["planner", "executor", "reflector", "generator", "verifier"]):
            vals = timings.get(node, [])
            avg_ms = round(sum(vals) / len(vals), 2) if vals else 0.0
            timing_cols[i].metric(node, f"{avg_ms} ms", f"{len(vals)} 次")

        if show_debug:
            with st.expander("查看 Agent 调试信息"):
                st.markdown("**检索步骤**")
                for step in result.get("steps", []):
                    st.write(f"- {step}")

                debug_tabs = st.tabs(["检索轨迹", "子问题", "校验", "耗时"])
                with debug_tabs[0]:
                    for item in result.get("retrieval_trace", []):
                        st.code(str(item))
                    if result.get("chapter_candidates"):
                        st.markdown("**chapter_candidates**")
                        st.code(str(result.get("chapter_candidates")))
                with debug_tabs[1]:
                    if result.get("sub_questions"):
                        st.markdown("**sub_questions**")
                        for q in result.get("sub_questions", []):
                            st.write(f"- {q}")
                    if result.get("sub_question_results"):
                        st.markdown("**sub_question_results**")
                        for item in result.get("sub_question_results", []):
                            st.code(str(item))
                with debug_tabs[2]:
                    for key in ["planner_raw", "reflector_raw", "verifier_raw", "verifier_feedback", "verifier_feedback_tag", "verifier_tag_history"]:
                        if result.get(key):
                            st.markdown(f"**{key}**")
                            st.code(str(result[key]))
                with debug_tabs[3]:
                    if result.get("node_timings_ms"):
                        st.code(str(result["node_timings_ms"]))

    st.session_state.messages.append({"role": "assistant", "content": answer})
