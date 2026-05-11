"""Structured output utility with native/fallback strategy.

Works across all LLM providers.
- GPT, Cerebras: native with_structured_output
- DeepSeek, Gemini: try native, fall back to JSON
- Ollama: always JSON fallback (no native support)
"""

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger("structured-output")

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when structured output fails after all retries."""


async def structured_invoke(
    model,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    max_retries: int = 2,
    config: dict | None = None,
) -> T:
    """Get structured output from an LLM with automatic native/fallback.

    Strategy:
    1. Try native with_structured_output() (GPT, Gemini, newer DeepSeek)
    2. Fall back to JSON prompting + parsing (Ollama, older models)
    3. Retry on parse failure with correction hint

    Args:
        model: LangChain chat model instance
        messages: List of messages to send
        schema: Pydantic model class for the expected output
        max_retries: Max retries for JSON fallback path (default 2)

    Returns:
        Validated Pydantic model instance

    Raises:
        StructuredOutputError: If all attempts (native + retries) fail
    """
    schema_name = schema.__name__
    model_name = _get_model_name(model)
    logger.debug(
        "[structured_output] ENTER | schema=%s | msgs=%d | model=%s | max_retries=%d",
        schema_name, len(messages), model_name, max_retries,
    )

    # ── Attempt native structured output ──
    native_available = hasattr(model, "with_structured_output")
    logger.debug(
        "[structured_output] native_available=%s | model=%s",
        native_available, model_name,
    )

    if native_available:
        try:
            logger.debug(
                "[structured_output] Trying NATIVE path | schema=%s", schema_name
            )
            structured_model = model.with_structured_output(schema)
            result = await structured_model.ainvoke(messages, config=config or {})
            if result is not None:
                logger.debug(
                    "[structured_output] NATIVE OK | schema=%s | type=%s",
                    schema_name, type(result).__name__,
                )
                return result
            else:
                logger.warning(
                    "[structured_output] NATIVE returned None | schema=%s | falling back to JSON",
                    schema_name,
                )
        except (NotImplementedError, AttributeError, TypeError) as e:
            logger.debug(
                "[structured_output] NATIVE not supported | schema=%s | err=%s",
                schema_name, e,
            )
        except Exception as e:
            logger.warning(
                "[structured_output] NATIVE failed | schema=%s | err=%s | falling back to JSON",
                schema_name, e,
            )

    # ── JSON fallback path ──
    logger.debug(
        "[structured_output] Switching to JSON_FALLBACK | schema=%s", schema_name
    )
    return await _invoke_with_json_fallback(model, messages, schema, max_retries, config or {})


async def _invoke_with_json_fallback(
    model,
    messages: list[BaseMessage],
    schema: type[T],
    max_retries: int,
    config: dict,
) -> T:
    """Prompt the model for JSON, parse and validate with retries."""
    schema_name = schema.__name__

    schema_json = schema.model_json_schema()
    schema_str = json.dumps(schema_json, indent=2, ensure_ascii=False)

    json_instruction = (
        "You must respond with ONLY valid JSON that matches this schema:\n"
        f"```json\n{schema_str}\n```\n"
        "Do NOT include markdown fences, explanations, or any other text. "
        "Output ONLY the JSON object."
    )

    augmented = list(messages) + [HumanMessage(content=json_instruction)]

    last_error = None
    last_text = ""
    for attempt in range(max_retries + 1):
        logger.debug(
            "[structured_output] JSON_FALLBACK attempt=%d/%d | schema=%s",
            attempt + 1, max_retries + 1, schema_name,
        )
        try:
            response = await model.ainvoke(augmented, config=config)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            last_text = text

            logger.debug(
                "[structured_output] JSON_FALLBACK response | len=%d | preview=%.120s",
                len(text) if text else 0, text if text else "<empty>",
            )

            if not text or not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response from model")

            cleaned = _extract_json(text)
            if cleaned != text:
                logger.debug(
                    "[structured_output] JSON_FALLBACK stripped markdown fence | "
                    "before=%d chars | after=%d chars",
                    len(text), len(cleaned),
                )

            data = json.loads(cleaned)
            validated = schema.model_validate(data)
            logger.debug(
                "[structured_output] JSON_FALLBACK OK | schema=%s | attempt=%d",
                schema_name, attempt + 1,
            )
            return validated

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            logger.warning(
                "[structured_output] JSON_FALLBACK attempt %d/%d FAILED | schema=%s | err=%s",
                attempt + 1, max_retries + 1, schema_name, e,
            )

            if attempt < max_retries:
                correction = (
                    f"Your previous response was invalid. Error: {str(e)}. "
                    "Please output ONLY valid JSON matching the schema. "
                    "Do NOT include markdown fences or extra text."
                )
                augmented.append(AIMessage(content=last_text))
                augmented.append(HumanMessage(content=correction))
                logger.debug(
                    "[structured_output] JSON_FALLBACK retry appended correction hint"
                )

    logger.error(
        "[structured_output] EXHAUSTED | schema=%s | attempts=%d | last_error=%s",
        schema_name, max_retries + 1, last_error,
    )
    raise StructuredOutputError(
        f"Failed to get valid structured output after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract raw JSON from text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _get_model_name(model) -> str:
    """Extract model name/identifier for logging."""
    for attr in ("model_name", "model"):
        val = getattr(model, attr, None)
        if val:
            return str(val)
    return type(model).__name__
