from utils.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from core.config import config

logger = get_logger("llm_factory")

_model_cache: dict[str, BaseChatModel] = {}
_bound_cache: dict[tuple, BaseChatModel] = {}

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def _create_gpt_model() -> BaseChatModel:
    return ChatOpenAI(
        model=config.CLOUD_CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def _create_cerebras_model() -> BaseChatModel:
    if not config.CEREBRAS_API_KEY:
        logger.warning("CEREBRAS_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return ChatOpenAI(
        model=config.CEREBRAS_CHAT_MODEL,
        base_url=CEREBRAS_BASE_URL,
        api_key=config.CEREBRAS_API_KEY,
    )


def _create_gemini_model() -> BaseChatModel:
    if not config.GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
    except ImportError:
        logger.warning("langchain-google-genai not installed, falling back to GPT")
        return _create_gpt_model()
    except Exception as e:
        logger.warning(f"Failed to create Gemini model: {e}, falling back to GPT")
        return _create_gpt_model()


def _create_ollama_model() -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.LOCAL_MODEL_NAME,
            base_url=config.OLLAMA_BASE_URL,
        )
    except ImportError:
        logger.warning("langchain-ollama not installed, falling back to GPT")
        return _create_gpt_model()


_factories = {
    "gpt-cloud": _create_gpt_model,
    "cerebras": _create_cerebras_model,
    "gemini": _create_gemini_model,
    "gemma-local": _create_ollama_model,
}


def get_model(selection: str = "gpt-cloud") -> BaseChatModel:
    """Unified factory for LLM instances with caching."""
    if selection not in _model_cache:
        factory = _factories.get(selection)
        if factory:
            _model_cache[selection] = factory()
            model_name = getattr(_model_cache[selection], "model_name", None) or getattr(_model_cache[selection], "model", "?")
            logger.info(f"Initialized model [{selection}]: {model_name}")
        else:
            logger.warning(f"Unknown model selection '{selection}', falling back to gpt-cloud")
            _model_cache[selection] = _create_gpt_model()

    return _model_cache[selection]


def get_bound_model(selection: str, tool_names: frozenset, all_tools: list) -> BaseChatModel:
    """Get a model with tools pre-bound. Caches by (selection, tool_names) to
    avoid re-binding on every call — .bind_tools() creates schemas each time."""
    key = (selection, tool_names)
    if key not in _bound_cache:
        base = get_model(selection)
        if tool_names:
            tools = [t for t in all_tools if t.name in tool_names]
            _bound_cache[key] = base.bind_tools(tools)
        else:
            _bound_cache[key] = base
    return _bound_cache[key]
