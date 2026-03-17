from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from src.core.config import get_settings


@lru_cache
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
        streaming=True,
    )
