"""Provider-reported usage, separate from local context estimates."""


def model_usage_metrics(response, model: str, elapsed_ms: int) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    raw = metadata.get("token_usage") or metadata.get("usage") or {}
    details = usage.get("input_token_details") or {}
    raw_details = raw.get("prompt_tokens_details") or {}

    def reported(*values):
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return {
        "event": "context.model_usage", "model": model,
        "input_tokens": reported(usage.get("input_tokens"), raw.get("prompt_tokens")),
        "output_tokens": reported(usage.get("output_tokens"), raw.get("completion_tokens")),
        "cache_read_tokens": reported(details.get("cache_read"), raw.get("prompt_cache_hit_tokens"), raw_details.get("cached_tokens")),
        "call_elapsed_ms": elapsed_ms,
        "source": "provider_reported; missing values are null; elapsed is not TTFT",
    }
