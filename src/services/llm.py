import asyncio

from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from utils.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from core.config import config
from services.mimo_chat import MiMoChatOpenAI

logger = get_logger("llm_factory")

_model_cache: dict[str, BaseChatModel] = {}
_bound_cache: dict[tuple, BaseChatModel] = {}

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


def _create_gpt_model(**client_options) -> BaseChatModel:
    return ChatOpenAI(
        model=config.CLOUD_CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        streaming=True,
        use_responses_api=True,
        **client_options,
    )


def _create_cerebras_model(**client_options) -> BaseChatModel:
    if not config.CEREBRAS_API_KEY:
        logger.warning("CEREBRAS_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return ChatOpenAI(
        model=config.CEREBRAS_CHAT_MODEL,
        base_url=CEREBRAS_BASE_URL,
        api_key=config.CEREBRAS_API_KEY,
        streaming=True,
        **client_options,
    )


def _create_deepseek_model(**client_options) -> BaseChatModel:
    if not config.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return ChatOpenAI(
        model=config.DEEPSEEK_FLASH_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        streaming=True,
        **client_options,
    )


def _create_deepseek_pro_model(**client_options) -> BaseChatModel:
    if not config.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return ChatOpenAI(
        model=config.DEEPSEEK_PRO_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        streaming=True,
        **client_options,
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


def _create_mimo_model(**client_options) -> BaseChatModel:
    if not config.MIMO_API_KEY:
        logger.warning("MIMO_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return MiMoChatOpenAI(
        model=config.MIMO_CHAT_MODEL,
        base_url=MIMO_BASE_URL,
        api_key=config.MIMO_API_KEY,
        streaming=True,
        stream_usage=True,
        extra_body={"thinking": {"type": "enabled"}},
        **client_options,
    )


def _create_mimo_summary_model(**client_options) -> BaseChatModel:
    """Independent summary model; changing it does not change chat or memory."""
    if not config.MIMO_API_KEY:
        raise ValueError("MIMO_API_KEY is required for context summaries")
    return MiMoChatOpenAI(
        model=config.MIMO_SUMMARY_MODEL,
        base_url=MIMO_BASE_URL,
        api_key=config.MIMO_API_KEY,
        streaming=True,
        **client_options,
    )


def _create_mimo_title_model(**client_options) -> BaseChatModel:
    """Explicit lightweight model for thread titles."""
    if not config.MIMO_API_KEY:
        raise ValueError("MIMO_API_KEY is required for thread titles")
    return MiMoChatOpenAI(
        model=config.MIMO_TITLE_MODEL,
        base_url=MIMO_BASE_URL,
        api_key=config.MIMO_API_KEY,
        streaming=True,
        **client_options,
    )


def _create_mimo_v25_model(**client_options) -> BaseChatModel:
    """Legacy selector, retained for existing non-title integrations."""
    if not config.MIMO_API_KEY:
        logger.warning("MIMO_API_KEY not set, falling back to GPT")
        return _create_gpt_model()
    return ChatOpenAI(
        model="mimo-v2.6", base_url=MIMO_BASE_URL,
        api_key=config.MIMO_API_KEY, streaming=True,
        **client_options,
    )


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
    "deepseek": _create_deepseek_model,
    "deepseek-pro": _create_deepseek_pro_model,
    "mimo": _create_mimo_model,
    "mimo-summary": _create_mimo_summary_model,
    "mimo-title": _create_mimo_title_model,
    "mimo-v2.6": _create_mimo_v25_model,
}


def get_strict_model(selection: str, *, isolated_http_clients: bool = False) -> BaseChatModel:
    """Explicit provider; closable short-lived models must own their HTTP pools."""
    keys = {
        "gpt-cloud": "OPENAI_API_KEY", "cerebras": "CEREBRAS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY", "deepseek-pro": "DEEPSEEK_API_KEY",
        "mimo": "MIMO_API_KEY", "mimo-summary": "MIMO_API_KEY", "mimo-title": "MIMO_API_KEY",
        "mimo-v2.6": "MIMO_API_KEY", "gemini": "GOOGLE_API_KEY",
    }
    if selection not in _factories:
        raise ValueError(f"Unknown model selection: {selection}")
    if selection in keys and not getattr(config, keys[selection]):
        raise ValueError(f"{keys[selection]} is required for {selection}")
    if selection == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_CHAT_MODEL, google_api_key=config.GOOGLE_API_KEY, max_retries=0)
    if selection == "gemma-local":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.LOCAL_MODEL_NAME, base_url=config.OLLAMA_BASE_URL)
    client_options = {}
    if isolated_http_clients:
        # New SDK instances otherwise still share LangChain's cached httpx
        # transports. Closing one would poison main/summary/title models.
        client_options = {"http_client": DefaultHttpxClient(),
                          "http_async_client": DefaultAsyncHttpxClient()}
    try:
        model = _factories[selection](**client_options)
    except BaseException:
        if client_options:
            client_options["http_client"].close()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(client_options["http_async_client"].aclose())
            else:
                loop.create_task(client_options["http_async_client"].aclose())
        raise
    model.max_retries = 0
    # Factories construct SDK clients eagerly; changing only the LangChain
    # field does not change the SDK's already-initialized retry setting.
    model.root_client.max_retries = 0
    model.root_async_client.max_retries = 0
    return model


def output_limit_kwargs(selection: str, tokens: int) -> dict:
    """Provider-specific output cap matching the context planner's reserve."""
    if selection == "gemini":
        return {"max_output_tokens": tokens}
    if selection == "gemma-local":
        return {"num_predict": tokens}
    if selection == "gpt-cloud":
        return {"max_completion_tokens": tokens}
    if selection in _factories:
        return {"max_tokens": tokens}
    raise ValueError(f"Unknown model selection: {selection}")


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


async def get_structured_response(
    selection: str,
    messages: list,
    schema: type,
    *,
    max_retries: int = 2,
    config: dict | None = None,
):
    """Get a structured (Pydantic) response from the specified LLM.

    Convenience wrapper that combines model factory lookup with
    structured_invoke for auto fallback across providers.
    """
    from utils.structured_output import structured_invoke

    logger.debug(
        "[get_structured_response] ENTER | model=%s | schema=%s | msgs=%d | retries=%d",
        selection, schema.__name__, len(messages), max_retries,
    )
    model = get_model(selection)
    try:
        result = await structured_invoke(model, messages, schema, max_retries=max_retries, config=config)
        logger.debug(
            "[get_structured_response] OK | model=%s | schema=%s | result_type=%s",
            selection, schema.__name__, type(result).__name__,
        )
        return result
    except Exception as e:
        logger.error(
            "[get_structured_response] FAILED | model=%s | schema=%s | err=%s",
            selection, schema.__name__, e,
        )
        raise


def get_bound_model(selection: str, tool_names: frozenset, all_tools: list) -> BaseChatModel:
    """Get a model with tools pre-bound. Caches by (selection, tool_names) to
    avoid re-binding on every call — .bind_tools() creates schemas each time."""
    key = (selection, tool_names)
    if key not in _bound_cache:
        # Selected main models never silently become another provider.
        # No SDK retry: one observable outcome for each request.
        base = get_strict_model(selection)
        if tool_names:
            tools = [t for t in all_tools if t.name in tool_names]
            _bound_cache[key] = base.bind_tools(tools)
        else:
            _bound_cache[key] = base
    return _bound_cache[key]
