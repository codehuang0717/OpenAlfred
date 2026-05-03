# src/llm.py
from utils.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from core.config import config

logger = get_logger("llm_factory")

_model_cache: dict[str, BaseChatModel] = {}

def get_model(selection: str = "gpt-cloud") -> BaseChatModel:
    """ Unified factory for LLM instances with caching. """
    if selection not in _model_cache:
        if selection == "gemma-local":
            try:
                from langchain_ollama import ChatOllama
                _model_cache[selection] = ChatOllama(
                    model=config.LOCAL_MODEL_NAME,
                    base_url=config.OLLAMA_BASE_URL,
                )
                logger.info(f"Initialized local Ollama model: {config.LOCAL_MODEL_NAME}")
            except ImportError:
                logger.warning("langchain-ollama not installed, falling back to cloud GPT")
                _model_cache[selection] = ChatOpenAI(model=config.CLOUD_CHAT_MODEL)
                logger.info(f"Initialized fallback cloud model: {config.CLOUD_CHAT_MODEL}")
        else:
            # Default to cloud model
            _model_cache[selection] = ChatOpenAI(model=config.CLOUD_CHAT_MODEL)
            logger.info(f"Initialized cloud GPT model: {config.CLOUD_CHAT_MODEL}")
            
    return _model_cache[selection]
