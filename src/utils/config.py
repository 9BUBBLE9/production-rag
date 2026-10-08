from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    telegram_bot_token: str = ""

    hf_api_key: str = ""
    deepseek_api_key: str = ""
    yandex_api_key: str = ""
    yandex_folder_id: str = ""

    active_llm_provider: Literal["ollama", "huggingface", "deepseek", "yandex"] = "ollama"

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"

    chunk_size: int = 500
    chunk_overlap: int = 50

    log_level: str = "INFO"


settings = Settings()