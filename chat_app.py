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
    """缓存友好调用层：缓存可复用的 run_agent 引用。"""
    return run_agent


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = []
    if "processed_summary" not in st.session_state:
        st.session_state.processed_summary = "未处理任何文件"


init_state()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    return text.strip()


def _decode_bytes_with_fallback(data: bytes) -> str:
    # 常见中英文编码兜底
    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gb18030",
        "gbk",
        "big5",
        "cp950",
        "cp936",
        "latin1",
    ]
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
        pages = []
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paras)

    if name.endswith(".csv"):
        text = _decode_bytes_with_fallback(data)
        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join([c.strip() for c in row if c is not None]) for row in reader]
        rows = [r for r in rows if r.strip()]
        return "\n".join(rows)

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
        text = _decode_bytes_with_fallback(data)
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text("\n")

    # 未知格式兜底：尽力按文本解码
    return _decode_bytes_with_fallback(data)


def chunk_text(text: str, max_chars: int = 320, overlap: int = 80) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
            continue

        start = 0
        step = max_chars - overlap
        while start < len(p):
            c = p[start : start + max_chars].strip()
            if c:
                chunks.append(c)
            if start + max_chars >= len(p):
                break
            start += step

    return chunks


def _chunk_unique_id(source: str, text: str) -> str:
    return hashlib.md5(f"{source}::{text}".encode("utf-8")).hexdigest()


def _normalize_existing_chunks(raw_chunks: List[dict]) -> List[dict]:
    """兼容旧索引格式：补齐 uid/source 字段并去除非法项。"""
    normalized = []
    for item in raw_chunks:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        source = (item.get("source") or "unknown").strip() or "unknown"
        uid = item.get("uid") or _chunk_unique_id(source, text)
        normalized.append({"uid": uid, "text": text, "source": source})
    return normalized


def save_uploaded_chunks_to_index(
    files,
    mode: str = "incremental",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """将上传文件切分后写入 index/novel_chunks.json，并重建 ES/Qdrant 索引。"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = os.path.join(INDEX_DIR, "novel_chunks.json")

    existing_chunks = []
    if mode == "incremental" and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fr:
            existing_chunks = _normalize_existing_chunks(json.load(fr))

    existing_ids = {item.get("uid") for item in existing_chunks if item.get("uid")}
    docs = []
    new_chunks = []

    total_files = len(files)
    for i, f in enumerate(files, start=1):
        if on_progress:
            on_progress(i - 1, total_files, f"正在解析与切分：{f.name}")

        raw = read_uploaded_file(f)
        chunks = chunk_text(raw)
        added_for_doc = 0

        for c in chunks:
            uid = _chunk_unique_id(f.name, c)
            if mode == "incremental" and uid in existing_ids:
                continue
            new_chunks.append({"uid": uid, "text": c, "source": f.name})
            existing_ids.add(uid)
            added_for_doc += 1

        docs.append(
            {
                "name": f.name,
                "size": f.size,
                "chunk_count": len(chunks),
                "added_chunk_count": added_for_doc,
            }
        )

        if on_progress:
            on_progress(i, total_files, f"已完成解析：{f.name}（{i}/{total_files}）")

    if mode == "overwrite":
        final_chunks = new_chunks
    else:
        final_chunks = existing_chunks + new_chunks

    # 为检索引擎分配稳定整型 ID（Qdrant/ES _id 兼容）
    chunk_data = []
    for idx, item in enumerate(final_chunks):
        chunk_data.append(
            {
                "id": idx,
                "uid": item["uid"],
                "text": item["text"],
                "source": item["source"],
            }
        )

    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(chunk_data, fw, ensure_ascii=False, indent=2)

    # 重建双路索引
    build_vector_store()
    build_bm25_index()

    return {
        "docs": docs,
        "new_chunks": len(new_chunks),
        "total_chunks": len(chunk_data),
        "mode": mode,
        "output_path": out_path,
    }


def get_available_sources() -> List[str]:
    """读取已入库来源列表，用于提问过滤。"""
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


st.title("Week2 Agentic RAG 对话")
st.caption("已接入 Week2 Agent 后端；支持文件上传并手动触发处理")

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

    index_mode = st.radio(
        "索引模式",
        options=["incremental", "overwrite"],
        format_func=lambda x: "增量追加" if x == "incremental" else "覆盖重建",
        horizontal=True,
    )

    files = st.file_uploader(
        "上传文档（支持 txt/md/pdf/docx/csv/json/html）",
        type=["txt", "md", "pdf", "docx", "csv", "json", "html", "htm"],
        accept_multiple_files=True,
        disabled=st.session_state.processing,
    )

    process_btn = st.button(
        "开始处理文件",
        use_container_width=True,
        disabled=st.session_state.processing or not files,
    )

    if process_btn and files:
        st.session_state.processing = True
        progress = st.progress(0, text="准备开始处理...")

        try:
            parse_weight = 70

            def _on_parse_progress(done: int, total: int, message: str) -> None:
                ratio = (done / total) if total else 0
                percent = int(parse_weight * ratio)
                progress.progress(percent, text=message)

            result = save_uploaded_chunks_to_index(
                files,
                mode=index_mode,
                on_progress=_on_parse_progress,
            )

            progress.progress(85, text="正在重建向量索引（Qdrant）...")
            # build_vector_store 在函数内已执行，这里仅用于用户提示节奏

            progress.progress(95, text="正在重建 BM25 索引（Elasticsearch）...")
            # build_bm25_index 在函数内已执行，这里仅用于用户提示节奏

            progress.progress(100, text="处理完成")

            st.session_state.processed_docs = result["docs"]
            st.session_state.processed_summary = (
                f"模式: {'增量追加' if result['mode'] == 'incremental' else '覆盖重建'}；"
                f"本次新增 {result['new_chunks']} 个文本块，当前总计 {result['total_chunks']} 个；"
                f"已写入 {result['output_path']} 并重建索引"
            )
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
                st.write(f"- {d['name']} | {d['chunk_count']} chunks | {d['size']} bytes")

    source_options = get_available_sources()
    selected_sources = st.multiselect(
        "提问来源过滤（可选，多选）",
        options=source_options,
        default=[],
        disabled=st.session_state.processing,
        help="自动读取已入库 source 列表；不选表示全库检索",
    )

    if st.button("清空对话", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.messages = []
        st.rerun()


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
            result = agent_runner(
                user_input,
                max_steps=max_steps,
                fast_mode=fast_mode,
                source_filter=source_filter,
            )
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

        answer = result.get("answer") or "（无回答）"
        st.markdown(answer)

        timings = result.get("node_timings_ms", {})
        st.caption(f"总耗时：{total_ms} ms")

        timing_cols = st.columns(5)
        for i, node in enumerate(["planner", "executor", "reflector", "generator", "verifier"]):
            vals = timings.get(node, [])
            avg_ms = round(sum(vals) / len(vals), 2) if vals else 0.0
            timing_cols[i].metric(f"{node}", f"{avg_ms} ms", f"{len(vals)} 次")

        if show_debug:
            with st.expander("查看 Agent 调试信息"):
                st.markdown("**检索步骤**")
                for step in result.get("steps", []):
                    st.write(f"- {step}")

                st.markdown("**retrieval_trace**")
                for item in result.get("retrieval_trace", []):
                    st.code(str(item))

                if result.get("planner_raw"):
                    st.markdown("**planner_raw**")
                    st.code(result["planner_raw"])

                if result.get("reflector_raw"):
                    st.markdown("**reflector_raw**")
                    st.code(result["reflector_raw"])

                if result.get("verifier_raw"):
                    st.markdown("**verifier_raw**")
                    st.code(result["verifier_raw"])

                if result.get("verifier_feedback"):
                    st.markdown("**verifier_feedback**")
                    st.code(result["verifier_feedback"])

                if result.get("verifier_feedback_tag"):
                    st.markdown("**verifier_feedback_tag**")
                    st.code(result["verifier_feedback_tag"])

                if result.get("verifier_tag_history"):
                    st.markdown("**verifier_tag_history**")
                    st.code(str(result["verifier_tag_history"]))

                if result.get("node_timings_ms"):
                    st.markdown("**node_timings_ms（原始）**")
                    st.code(str(result["node_timings_ms"]))

    st.session_state.messages.append({"role": "assistant", "content": answer})
