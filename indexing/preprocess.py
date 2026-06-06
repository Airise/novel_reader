import json
import os
import re
import sys
from typing import List

# 允许以 `python indexing/preprocess.py` 方式直接运行
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    INDEX_DIR,
    TARGET_CHARS,
    TXT_FILE,
)
SENT_ENDINGS = "。！？；!?;"
SECONDARY_ENDINGS = "，,"


def load_txt():
    """鲁棒读取文本，兼容常见中文编码。"""
    encodings = ["utf-8", "utf-8-sig", "utf-16", "gb18030", "gbk"]
    last_error = None
    for enc in encodings:
        try:
            with open(TXT_FILE, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"无法解码小说文件，请确认编码。尝试过编码: {encodings}，最后错误: {last_error}"
    )


def normalize_text(text: str) -> str:
    """归一化换行与空白字符"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_sentence(text: str) -> List[str]:
    """按句号等中文标点切句，保留标点。"""
    parts = re.split(r"([。！？；!?;])", text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        if not sent:
            continue
        if i + 1 < len(parts):
            sent += parts[i + 1]
        sentences.append(sent)
    return sentences


def find_best_cut(text: str, start: int, max_chars: int) -> int:
    """优先在句末符号处断开，其次逗号，最后硬切。"""
    hard_end = min(len(text), start + max_chars)
    window = text[start:hard_end]

    for idx in range(len(window) - 1, -1, -1):
        if window[idx] in SENT_ENDINGS:
            return start + idx + 1

    for idx in range(len(window) - 1, -1, -1):
        if window[idx] in SECONDARY_ENDINGS:
            return start + idx + 1

    return hard_end


def split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """长文本切分：句子边界优先 + 滑窗 overlap。"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = find_best_cut(text, start, max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def merge_short_paragraphs(paragraphs: List[str]) -> List[str]:
    """短段拼接到目标长度附近，减少噪声 chunk。"""
    merged = []
    buf = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if not buf:
            buf = p
            continue

        if len(buf) < CHUNK_MIN_CHARS or len(buf) + len(p) <= TARGET_CHARS:
            buf = f"{buf}\n{p}".strip()
        else:
            merged.append(buf)
            buf = p

    if buf:
        merged.append(buf)

    return merged


def chunk_text(text: str) -> List[str]:
    """先段落合并，再对超长段按句子边界优先切分。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_paragraphs = merge_short_paragraphs(paragraphs)

    final_chunks = []
    for p in merged_paragraphs:
        if len(p) <= CHUNK_MAX_CHARS:
            final_chunks.append(p)
            continue

        # 超长段先切句再拼块，保持语义边界
        sentences = split_by_sentence(p)
        if not sentences:
            final_chunks.extend(
                split_long_text(
                    p, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS
                )
            )
            continue

        buf = ""
        for s in sentences:
            if len(buf) + len(s) <= CHUNK_MAX_CHARS:
                buf += s
            else:
                if buf:
                    final_chunks.append(buf.strip())
                buf = s

        if buf:
            final_chunks.append(buf.strip())

        # 兜底：如果仍有超长 chunk，则走断点优先滑窗切
        refined = []
        for c in final_chunks:
            if len(c) > CHUNK_MAX_CHARS:
                refined.extend(
                    split_long_text(
                        c, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS
                    )
                )
            else:
                refined.append(c)
        final_chunks = refined

    final_chunks = [c for c in final_chunks if c.strip()]

    # 末尾回并：最后一个 chunk 过短时，合并到前一个 chunk
    if len(final_chunks) >= 2 and len(final_chunks[-1]) < CHUNK_MIN_CHARS:
        merged_tail = f"{final_chunks[-2]}\n{final_chunks[-1]}".strip()
        if len(merged_tail) <= CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS:
            final_chunks[-2] = merged_tail
            final_chunks.pop()

    return final_chunks


def build_quality_report(chunks: List[str]):
    lengths = [len(c) for c in chunks]
    total = len(lengths)

    if total == 0:
        return {
            "total_chunks": 0,
            "message": "没有可用 chunk",
            "config": {
                "chunk_max_chars": CHUNK_MAX_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "chunk_min_chars": CHUNK_MIN_CHARS,
                "target_chars": TARGET_CHARS,
            },
            "distribution": {},
            "samples": {"short": [], "medium": [], "long": []},
        }

    sorted_lengths = sorted(lengths)

    def percentile(p: float) -> int:
        idx = int((total - 1) * p)
        return sorted_lengths[idx]

    def ratio(count: int) -> float:
        return round(count / total, 4)

    short_chunks = [c for c in chunks if len(c) < CHUNK_MIN_CHARS]
    medium_chunks = [c for c in chunks if CHUNK_MIN_CHARS <= len(c) <= CHUNK_MAX_CHARS]
    long_chunks = [c for c in chunks if len(c) > CHUNK_MAX_CHARS]

    report = {
        "total_chunks": total,
        "config": {
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
            "chunk_min_chars": CHUNK_MIN_CHARS,
            "target_chars": TARGET_CHARS,
        },
        "stats": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(sum(lengths) / total, 2),
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
        },
        "distribution": {
            f"< {CHUNK_MIN_CHARS}": {
                "count": len(short_chunks),
                "ratio": ratio(len(short_chunks)),
            },
            f"{CHUNK_MIN_CHARS} ~ {CHUNK_MAX_CHARS}": {
                "count": len(medium_chunks),
                "ratio": ratio(len(medium_chunks)),
            },
            f"> {CHUNK_MAX_CHARS}": {
                "count": len(long_chunks),
                "ratio": ratio(len(long_chunks)),
            },
        },
        "samples": {
            "short": [s[:220] for s in short_chunks[:3]],
            "medium": [s[:220] for s in medium_chunks[:3]],
            "long": [s[:220] for s in long_chunks[:3]],
        },
    }

    return report


def main():
    text = normalize_text(load_txt())
    chunks = chunk_text(text)
    chunk_data = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

    output_path = os.path.join(INDEX_DIR, "novel_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    report = build_quality_report(chunks)
    report_path = os.path.join(INDEX_DIR, "chunk_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"共切分 {len(chunk_data)} 个文本块，保存至 {output_path}")
    print(f"切分质量报告已保存至 {report_path}")


if __name__ == "__main__":
    main()