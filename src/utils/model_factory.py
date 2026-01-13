"""模型工厂类"""
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


def get_model(name, provider, temperature=0.7, timeout=30):
    """初始化并返回聊天模型实例"""
    model = init_chat_model(
        model=name,
        model_provider=provider,
        temperature=temperature,
        timeout=timeout
    )
    return model


def get_embedding(name, provider):
    """初始化并返回嵌入模型实例"""
    embedding = init_embeddings(
        name,
        provider=provider
    )
    return embedding


def custom_model(name, provider, base_url, api_key, temperature=0.7, timeout=30):
    """初始化并返回聊天模型实例(openrouter专用)"""
    model = init_chat_model(
        model=name,
        model_provider=provider,
        temperature=temperature,
        timeout=timeout,
        base_url=base_url,
        api_key=api_key
    )
    return model


def custom_embedding(name, provider, base_url, api_key):
    embedding = init_embeddings(
        name,
        provider=provider,
        base_url=base_url,
        api_key=api_key
    )
    return embedding
