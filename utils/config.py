import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# 小说文件
TXT_FILE = os.path.join(DATA_DIR, "novel.txt")

# 向量库配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "novel_chunks"
VECTOR_SIZE = 1024  # bge-large-zh 维度

# Elasticsearch 配置（jieba 预分词 + standard 分词器）
ES_HOST = "http://localhost:9200"
ES_INDEX = "novel_chunks"
ES_ANALYZER = "standard"

# 检索参数（实验A）
RETRIEVAL_K = 30       # 每种检索器召回数量
RRF_K = 60             # RRF 平滑参数
FUSION_TOP_N = 60      # RRF 融合后进入 rerank 的候选池大小
RERANK_TOP_K = 5       # 最终返回数量
RERANK_MIN_SCORE = 0.15  # rerank 最小相关性阈值（低于则丢弃）

# 预处理参数（实验B）
ENABLE_INCREMENTAL_INDEXING = True  # 是否启用章节级增量索引
CHAPTER_SPLIT_REGEX = r"(?m)^(?:\s*)(?:序章|楔子|尾声|第[一二三四五六七八九十百千万0-9]+[卷部篇章节回](?:\s*.*)?)\s*$"
CHUNK_MAX_CHARS = 280       # 单 chunk 最大字符数
CHUNK_OVERLAP_CHARS = 100   # 长段滑窗重叠字符数
CHUNK_MIN_CHARS = 80        # 过短段落拼接阈值
TARGET_CHARS = 200          # 段落拼接目标长度
CHUNK_HASH_ALGO = "md5"     # chunk / chapter 哈希算法

# 中文嵌入模型
EMBEDDING_MODEL = "BAAI/bge-large-zh"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-3f641299ceca4e3d88992d117a98018b"  # 请替换为你的 API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"