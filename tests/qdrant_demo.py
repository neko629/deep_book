import sys
import os
import qdrant_client
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 将项目根目录添加到路径，以便导入 config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from src import config


def run_qdrant_demo():
    print(f"--- 深度环境探测与现代化 API 测试 ---")

    # 1. 初始化客户端
    try:
        clean_host = config.QDRANT_HOST.replace("http://", "").replace("https://",
                                                                       "").strip("/")
        client = QdrantClient(host=clean_host, port=config.QDRANT_PORT)
        print(f"成功连接至 Qdrant: {clean_host}:{config.QDRANT_PORT}")
    except Exception as e:
        print(f"实例化失败: {e}")
        return

    # 2. 准备集合 (Collection)
    # 注意：如果要使用你找到的 models.Document 模式，Qdrant 建议使用现代化的接口
    collection_name = "demo_collection"
    try:
        # 重置集合，使用 4 维模拟向量进行基础测试
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )

        # 插入模拟数据
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"text": "萧炎来到了乌坦城", "chapter": "第一章"}
                )
            ]
        )
        print(f"集合已就绪并插入测试数据。")
    except Exception as e:
        print(f"基础操作失败: {e}")
        return

    # 3. 检索测试
    print("\n--- 执行检索测试 ---")

    # 方案 A：使用你找到的 query_points 接口 (这是 Qdrant 1.10+ 的标准写法)
    # 注意：只有在 Qdrant 服务端配置了嵌入模型时，models.Document 才能直接运行
    # 我们这里先演示如何用 query_points 配合已有的向量（最稳妥的工业做法）
    try:
        print("正在尝试现代化 query_points 检索...")
        search_result = client.query_points(
            collection_name=collection_name,
            query=[0.11, 0.21, 0.31, 0.41],  # 模拟经过 bge-m3 转化后的查询向量
            limit=1
        )

        if search_result.points:
            point = search_result.points[0]
            print(
                f"检索成功！找到内容: {point.payload.get('text')} (得分: {point.score:.4f})")
    except Exception as e:
        print(f"query_points 检索失败: {e}")

    # 4. 知识科普：关于你找到的 models.Document
    print("\n--- 导师说明 ---")
    print("你找到的代码片段：")
    print("query=models.Document(text='...', model='...')")
    print("\n这种方式被称为 'In-database Embedding'。")
    print("优点：代码极其简洁，不用自己在本地处理 bge-m3。")
    print("缺点：需要 Qdrant 服务器具备计算能力，且模型选择受限。")
    print(f"对于你的项目，由于使用了自定义的 {config.EMBEDDING_MODEL}，")
    print("我们建议：在 Python 中用 LangChain 调用模型产生向量，再传给 Qdrant。")


if __name__ == "__main__":
    run_qdrant_demo()