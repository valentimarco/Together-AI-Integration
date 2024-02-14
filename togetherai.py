from typing import List, Optional, Type
from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, ConfigDict, SecretStr
from datetime import datetime, date
from cat.factory.llm import LLMSettings
from langchain_openai.chat_models import ChatOpenAI


class CustomOpenAI(ChatOpenAI):

    def __init__(self, **kwargs):
        model_kwargs = {}

        super().__init__(
            openai_api_key=kwargs['api_key'],
            model_kwargs=model_kwargs,
            **kwargs
        )

        self.openai_api_base = kwargs['base_url']


class TogetherAIConfig(LLMSettings):
    api_key: str
    base_url: str = "https://api.together.xyz/v1"
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True

    _pyclass: Type = CustomOpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "TogetherAI",
            "description": "Configuration for TogetherAI",
            "link": "https://www.together.ai",
        }
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(TogetherAIConfig)
    return allowed
