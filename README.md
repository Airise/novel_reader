# Week2 Agentic RAG（中文小说问答 / 审校助手）

一个基于 **LangGraph + 混合检索（Qdrant 向量 + Elasticsearch BM25）+ Reranker** 的中文长文本问答项目，面向中文小说、剧情追踪与编辑审校场景。

当前项目已支持：

- 多轮 Agent 检索编排（`router / planner / executor / reflector / generator / verifier`）
- 问题类型识别（`fact / tracing / comparison / contradiction / causal / search`）
- 子问题拆解与多步检索
- 文件上传后自动切分与索引重建
- 增量追加 / 覆盖重建 两种索引模式
- 章节级切分、章节 hash、chunk hash
- 检索降级容错（Qdrant / Elasticsearch 任一故障时自动切换）
- 按来源（source）过滤提问
- Streamlit 可视化调试（步骤、检索轨迹、节点耗时、子问题结果、章节候选）

---

## 1. 项目结构

```text
rag/
├─ agent/
│  ├─ graph.py              # LangGraph 工作流
│  ├─ nodes.py              # router/planner/executor/reflector/generator/verifier 节点
│  └─ state.py              # AgentState 定义
├─ indexing/
│  ├─ preprocess.py         # 章节级切分、hash、chunk 结构化输出
│  ├─ build_vector_store.py # 构建 Qdrant 向量索引
│  └─ build_bm25_index.py   # 构建 Elasticsearch BM25 索引
├─ retrieval/
│  ├─ hybrid_retriever.py   # 混合检索 + RRF + rerank + 降级
│  └─ reranker.py
├─ utils/
│  └─ config.py             # 核心配置（端口、模型、检索参数、切分参数）
├─ index/
│  ├─ novel_chunks.json     # 持久化 chunks（含 chapter_id / chapter_title / hash）
│  ├─ chapter_hashes.json   # 章节 hash 清单
│  └─ chunk_quality_report.json
├─ prompts/
│  ├─ planner_prompt.txt
│  └─ reflector_prompt.txt
├─ chat_app.py              # Streamlit 前端
├─ test_agent.py            # 命令行测试入口
├─ requirements.txt
└─ problem.md               # 开发问题与修复记录
```

---

## 2. 核心能力

### 2.1 章节级结构化切分

项目不再只做固定长度切块，而是先按章节识别，再在章节内做段落 / 句子级聚合。chunk 会带上：

- `chapter_id`
- `chapter_title`
- `chapter_hash`
- `chunk_hash`
- `uid`

这对中文小说、连载长文、剧情追踪类问答更友好。

### 2.2 问题类型路由

Agent 会先识别问题类型，再决定检索与生成策略。当前支持：

- `fact`：事实问答
- `tracing`：情节追踪 / 追问
- `comparison`：对比分析
- `contradiction`：前后矛盾检测
- `causal`：原因分析 / 过程分析
- `search`：关键词定位

### 2.3 子问题拆解

对追踪、对比、矛盾、过程类问题，会自动生成子问题并分别检索，再合并证据进行生成与校验。

### 2.4 检索降级

当 Qdrant 或 Elasticsearch 任一服务不可用时，系统会自动降级为单路检索，避免整条问答链路直接失败。

### 2.5 可观测性

Streamlit 前端支持查看：

- 检索步骤 `steps`
- `retrieval_trace`
- 子问题列表 `sub_questions`
- 子问题结果 `sub_question_results`
- 章节候选 `chapter_candidates`
- planner / reflector / verifier 原始输出
- 各节点耗时

---

## 3. 环境要求

- Python 3.10+（建议 3.12）
- Qdrant（默认 `localhost:6333`）
- Elasticsearch（默认 `http://localhost:9200`）
- Windows / macOS / Linux 均可

---

## 4. 安装与启动

### 4.1 创建并激活虚拟环境（Windows PowerShell）

```bash
cd y:\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.2 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 4.3 启动依赖服务

可使用 Docker 快速启动：

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.13.4
```

### 4.4 启动前端

```bash
streamlit run chat_app.py
```

浏览器访问终端输出的本地地址（通常是 `http://localhost:8501`）。

---

## 5. 配置说明

主要配置文件：`utils/config.py`

重点参数：

- 向量检索：`QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_COLLECTION`
- BM25 检索：`ES_HOST`、`ES_INDEX`
- 检索参数：`RETRIEVAL_K`、`RRF_K`、`FUSION_TOP_N`、`RERANK_TOP_K`
- 分块参数：`CHUNK_MAX_CHARS`、`CHUNK_OVERLAP_CHARS`、`CHUNK_MIN_CHARS`、`TARGET_CHARS`
- 章节识别：`CHAPTER_SPLIT_REGEX`
- LLM 参数：`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY`

> `DEEPSEEK_API_KEY` 当前仍为便捷写法，若要部署到公共环境，建议改为环境变量读取。

---

## 6. 使用方式

### 6.1 Web（推荐）

1. 打开 `chat_app.py` 页面
2. 在侧栏选择索引模式：
   - `增量追加`：追加新 chunk，按 `uid` 去重
   - `覆盖重建`：只保留本次上传内容
3. 上传文件（`txt / md / pdf / docx / csv / json / html`）
4. 点击“开始处理文件”
5. 提问（可选来源过滤）

### 6.2 命令行调试

```bash
python test_agent.py --max-steps 4 --debug
```

可选参数：

- `--max-steps`：最大检索步数
- `--fast-mode`：快速模式（减少回环）
- `--debug`：输出 planner / reflector / verifier 原始信息与 trace

---

## 7. 索引与增量策略

### 7.1 当前数据结构

- `index/novel_chunks.json`：chunk 持久化文件
- `index/chapter_hashes.json`：章节 hash 清单
- `index/chunk_quality_report.json`：切分质量报告

### 7.2 chunk 唯一标识

- `uid = md5(source::text)`

### 7.3 章节级元数据

chunk 会写入：

- `chapter_id`
- `chapter_title`
- `chapter_hash`
- `chunk_hash`

### 7.4 重建策略

每次处理后会重建：

- Qdrant 向量索引
- Elasticsearch BM25 索引

> 当前版本已经为章节级增量更新预留了数据结构，后续可进一步做到“只重建变化章节”。

---

## 8. 性能优化建议（当前项目可直接调）

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

## 9. 常见问题

### Q1：上传时报 `KeyError: 'uid'`

旧索引格式导致。当前版本已做兼容迁移，重试处理即可。

### Q2：`No module named ...`

请确认已激活项目 `.venv`，并用 `python -m pip` 安装依赖。

### Q3：连接不上 ES / Qdrant

先确认容器已启动且端口与 `utils/config.py` 一致。

### Q4：为什么有些小说标题切分不准？

当前采用的是通用章节识别规则，支持 `卷 / 部 / 篇 / 章 / 节 / 回` 等常见格式。若你的文本标题风格特殊，可进一步扩展 `CHAPTER_SPLIT_REGEX`。

---

## 10. 开发记录

详细问题与修复过程见：`problem.md`

包含：

- 环境依赖问题
- 编码 / 导入问题
- 检索质量与循环收敛问题
- 性能与可观测性改进记录
- 增量更新、检索降级、子问题拆解等改造记录

---

## 11. License

仅供学习与实验使用。
