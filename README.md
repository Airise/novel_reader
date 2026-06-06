# Week2 Agentic RAG（中文小说问答）

一个基于 **LangGraph + 混合检索（Qdrant 向量 + Elasticsearch BM25）+ Reranker** 的中文问答项目，支持：

- 多轮 Agent 检索规划（planner / executor / reflector / generator / verifier）
- 文件上传后自动切分与索引重建
- **增量追加 / 覆盖重建** 两种索引模式
- 按来源（source）过滤提问
- Streamlit 可视化调试（steps、retrieval trace、节点耗时）

---

## 1. 项目结构

```text
rag/
├─ agent/
│  ├─ graph.py            # LangGraph 工作流
│  ├─ nodes.py            # planner/executor/reflector/generator/verifier 节点
│  └─ state.py            # AgentState 定义
├─ indexing/
│  ├─ build_vector_store.py  # 构建 Qdrant 向量索引
│  └─ build_bm25_index.py    # 构建 Elasticsearch BM25 索引
├─ retrieval/
│  ├─ hybrid_retriever.py    # 混合检索 + RRF + rerank
│  └─ reranker.py
├─ utils/
│  └─ config.py           # 核心配置（端口、模型、检索参数、切分参数）
├─ index/
│  └─ novel_chunks.json   # 持久化 chunks（含 id/uid/text/source）
├─ prompts/
│  ├─ planner_prompt.txt
│  └─ reflector_prompt.txt
├─ chat_app.py            # Streamlit 前端
├─ test_agent.py          # 命令行测试入口
├─ requirements.txt
└─ problem.md             # 开发问题与修复记录
```

---

## 2. 环境要求

- Python 3.10+（建议 3.12）
- Qdrant（默认 `localhost:6333`）
- Elasticsearch（默认 `http://localhost:9200`）
- Windows / macOS / Linux 均可

---

## 3. 安装与启动

### 3.1 创建并激活虚拟环境（Windows PowerShell）

```bash
cd y:\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.2 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3.3 启动依赖服务

可使用 Docker 快速启动：

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.13.4
```

### 3.4 启动前端

```bash
streamlit run chat_app.py
```

浏览器访问终端输出的本地地址（通常是 `http://localhost:8501`）。

---

## 4. 配置说明

主要配置文件：`utils/config.py`

重点参数：

- 向量检索：`QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_COLLECTION`
- BM25 检索：`ES_HOST`、`ES_INDEX`
- 检索参数：`RETRIEVAL_K`、`RRF_K`、`FUSION_TOP_N`、`RERANK_TOP_K`
- 质量阈值：`RERANK_MIN_SCORE`
- 分块参数：`CHUNK_MAX_CHARS`、`CHUNK_OVERLAP_CHARS`、`CHUNK_MIN_CHARS`
- LLM 参数：`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY`

> 建议将 `DEEPSEEK_API_KEY` 改为环境变量读取，避免明文写入代码。

---

## 5. 使用方式

### 5.1 Web（推荐）

1. 打开 `chat_app.py` 页面
2. 在侧栏选择索引模式：
   - `增量追加`：只追加新 chunk（按 `uid` 去重）
   - `覆盖重建`：只保留本次上传内容
3. 上传文件（txt / md / pdf / docx / csv / json / html）
4. 点击“开始处理文件”
5. 提问（可选来源过滤）

### 5.2 命令行调试

```bash
python test_agent.py --max-steps 4 --debug
```

可选参数：

- `--max-steps`：最大检索步数
- `--fast-mode`：快速模式（减少回环）
- `--debug`：输出 planner/reflector/verifier 原始信息与 trace

---

## 6. 索引与增量策略

- chunk 唯一标识：`uid = md5(source::text)`
- `index/novel_chunks.json` 是单一持久化数据源
- 增量模式会自动兼容旧数据（旧记录缺少 `uid` 时会补齐）

每次处理后会重建：

- Qdrant 向量索引
- Elasticsearch BM25 索引

---

## 7. 性能优化建议（当前项目可直接调）

如果回答偏慢，可优先调整：

1. 降低召回规模
   - `RETRIEVAL_K`: 30 -> 15
   - `FUSION_TOP_N`: 60 -> 30
2. 降低 rerank 输出
   - `RERANK_TOP_K`: 5 -> 3
3. 开启前端快速模式
   - 跳过 verifier 二次回环，减少 LLM 调用
4. 调整分块
   - 适当增大 `CHUNK_MAX_CHARS`，减少 chunk 总数
5. 只在必要时使用来源过滤
   - 过滤后候选更少，通常更快

---

## 8. 常见问题

### Q1：上传时报 `KeyError: 'uid'`
旧索引格式导致。当前版本已做兼容迁移，重试处理即可。

### Q2：`No module named ...`
请确认已激活项目 `.venv`，并用 `python -m pip` 安装依赖。

### Q3：连接不上 ES / Qdrant
先确认容器已启动且端口与 `utils/config.py` 一致。

---

## 9. 开发记录

详细问题与修复过程见：`problem.md`

包含：
- 环境依赖问题
- 编码/导入问题
- 检索质量与循环收敛问题
- 性能与可观测性改进记录

---

## 10. License

仅供学习与实验使用。