import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载环境变量
# 这会读取项目根目录下的 .env 文件
load_dotenv()

# 2. 基础目录路径
# 定位到 nexus_reader/ 目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据相关路径
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"             # 存放原始小说
PROCESSED_DATA_DIR = DATA_DIR / "processed" # 存放清洗后的文本
DB_DIR = BASE_DIR / "database"              # 存放本地缓存

# 3. 模型配置 (从 .env 读取)
# 语言模型 (如 qwen2.5:7b)
LLM_MODEL = os.getenv("MODEL_NAME", "qwen2.5:7b")
# 向量模型 (如 bge-m3)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")

# Ollama 默认地址 (通常本地运行 Ollama 端口为 11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OPENROUTER 配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# 4. RAG 参数配置
# 针对几百万字小说，设置较大的分片以保持语境完整
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# 5. 企业级向量数据库 Qdrant 配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "novel_knowledge_gmzz"

# 6. 自动初始化必要的文件夹
def init_project_structure():
    """初始化项目所需的目录结构"""
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, DB_DIR]
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            # print(f"已创建目录: {d}")

# 执行初始化
init_project_structure()

if __name__ == "__main__":
    # 打印调试信息，确保配置读取正确
    print("-" * 30)
    print(f"NexusReader 配置检查:")
    print(f"项目路径: {BASE_DIR}")
    print(f"LLM模型: {LLM_MODEL}")
    print(f"Embedding模型: {EMBEDDING_MODEL}")
    print(f"Qdrant地址: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"切片设置: Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}")
    print("-" * 30)