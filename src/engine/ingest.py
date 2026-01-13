"""分割并灌注数据."""
import sys
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import models
from langchain_qdrant import QdrantVectorStore
import src.utils.model_factory as model_factory
from src import config
from langchain_text_splitters import RecursiveCharacterTextSplitter

class NovelIngestor:
    """
    小说数据灌注类
    先切分, 再转为向量存入数据库
    """

    def __init__(self):
        # 初始化嵌入模型
        self.embedding_model = model_factory.custom_embedding(
            name = "openai/text-embedding-3-small",
            provider = "openai",
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

        # 初始化 Qdrant 客户端
        self.qdrant_client = QdrantClient(
            host = config.QDRANT_HOST,
            port = config.QDRANT_PORT
        )

    def _init_collection(self, vector_size: int):
        """初始化 Collection, 如果存在就跳过"""
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == config.COLLECTION_NAME for c in collections)

        if not exists:
            print(f"创建新的 Collection: {config.COLLECTION_NAME}")
            self.qdrant_client.create_collection(
                collection_name = config.COLLECTION_NAME,
                vectors_config = models.VectorParams(
                    size = vector_size,
                    distance = models.Distance.COSINE
                )
            )
        else:
            print(f"集合 {config.COLLECTION_NAME} 已存在, 跳过创建.")

    def ingest(self, cleaned_file_path: str, batch_size: int = 50):
        """执行灌注流程."""
        file_path = Path(config.PROCESSED_DATA_DIR / cleaned_file_path)
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return

        # 读取并切分
        print(f"正在加载文件: {file_path.name}")
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.CHUNK_SIZE,
            chunk_overlap = config.CHUNK_OVERLAP
        )

        documents = text_splitter.create_documents(
            texts = [text],
            metadatas = [
                {
                    "source": file_path.name,
                    "ingest_time": time.time()
                }
            ]
        )

        total_docs = len(documents)
        print(f"文本已被分割为 {total_docs} 个片段")

        # 准备集合
        self._init_collection(1536)

        # 初始化 Vector store 包装器
        vector_store = QdrantVectorStore(
            client = self.qdrant_client,
            collection_name = config.COLLECTION_NAME,
            embedding = self.embedding_model
        )

        # 批量插入
        print(f"开始注入数据, 每批{batch_size}条")
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i +batch_size]
            try:
                vector_store.add_documents(batch)
                progress = min(i +batch_size, total_docs)
                print(f"✨ 进度: {progress}/{total_docs} ({(progress/total_docs)*100:.1f}%)")
            except Exception as e:
                print(f"批次{i}写入失败, 错误: {e}")
                time.sleep(2) # 2 秒后重试一次
                vector_store.add_documents(batch)

        print("✅ 数据灌注完成!")

if __name__ == "__main__":
    ingestor = NovelIngestor()
    ingestor.ingest("cleaned_gmzz.txt")
