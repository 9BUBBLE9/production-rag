from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from loguru import logger

from src.utils.config import settings


def get_llm(temperature: float = 0.0):
    """Возвращает LLM-клиент в зависимости от активного провайдера."""
    
    if settings.active_llm_provider == "ollama":
        logger.info(f"Инициализация локальной Ollama: {settings.ollama_model}")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
        
    elif settings.active_llm_provider == "huggingface":
        if not settings.hf_api_key:
            raise ValueError("HF_API_KEY не задан в .env")
        logger.info(f"Инициализация HuggingFace: {settings.hf_model_name}")
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model_name,
            huggingfacehub_api_token=settings.hf_api_key,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=llm_endpoint)

    elif settings.active_llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY не задан в .env")
        logger.info(f"Инициализация DeepSeek: {settings.deepseek_model}")
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )
        
    elif settings.active_llm_provider == "yandex":
        if not settings.yandex_api_key:
            raise ValueError("YANDEX_API_KEY не задан в .env")
        logger.info(f"Инициализация YandexGPT: {settings.yandex_model}")
        return ChatOpenAI(
            model=settings.yandex_model,
            api_key=settings.yandex_api_key,
            base_url="https://llm.api.cloud.yandex.ru/foundationModels/v1/openai/v1",
            temperature=temperature,
        )
    else:
        raise ValueError(f"Неизвестный провайдер: {settings.active_llm_provider}")