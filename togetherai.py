from typing import List, Optional, Type
from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, ConfigDict, SecretStr
from datetime import datetime, date
from cat.factory.llm import LLMSettings
from langchain_together import ChatTogether

class TogetherAIConfig004(LLMSettings):
    together_api_key: str
    together_api_base: str = "https://api.together.xyz/v1"
    model_name: str = "meta-llama/Llama-3-8b-chat-hf"
    temperature: float = 0.7
    streaming: bool = True

    _pyclass: Type = ChatTogether

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "TogetherAI",
            "description": "Configuration for TogetherAI",
            "link": "https://www.together.ai",
        }
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(TogetherAIConfig004)
    return allowed
