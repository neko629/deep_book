from langchain.agents import create_agent
from qdrant_client import QdrantClient
from src import config
import src.utils.model_factory as model_factory

model = model_factory.get_model(
    "deepseek-chat", "deepseek"
)

embedding = model_factory.custom_embedding(
            name = "openai/text-embedding-3-small",
            provider = "openai",
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

agent = create_agent(
    model = model,
    system_prompt = """
    你是一个小说内容问答机器人, 会根据用户提供给你的内容总结出答案, 如果用户提供的内容里无法得出答案, 请回答不知道
    """
)

qdrant_client = QdrantClient(
    host = config.QDRANT_HOST,
    port = config.QDRANT_PORT,
)

query = ("奥黛丽的性格是怎么样的")

embedding_query = embedding.embed_query(query)

search_result = qdrant_client.query_points(
            collection_name = config.COLLECTION_NAME,
            query = embedding_query,
            limit = 50
)

user_query = f"我想知道: {query}, 可以提供给你以下的信息: {search_result}"

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": user_query}]
    }
)
print(result["messages"][-1])


