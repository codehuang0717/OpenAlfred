import time
import jwt as pyjwt
from core.config import config


class MissingUserContextError(RuntimeError):
    """Raised when an authenticated user cannot be resolved for an operation."""


class UserContextMismatchError(RuntimeError):
    """Raised when request metadata contains conflicting user identities."""


class MissingThreadContextError(RuntimeError):
    """Raised when a graph operation has no concrete thread identifier."""


def _config_mapping(config) -> dict:
    if isinstance(config, dict):
        return config
    if config is None:
        return {}
    configurable = getattr(config, "configurable", None)
    metadata = getattr(config, "metadata", None)
    result = {}
    if isinstance(configurable, dict):
        result["configurable"] = configurable
    if isinstance(metadata, dict):
        result["metadata"] = metadata
    return result


def require_user_id(config) -> str:
    """Resolve one non-default user identity or fail closed.

    All accepted fields are trusted transport metadata populated by LangGraph auth
    or the voice service. Multiple fields may be present, but they must agree.
    """
    mapping = _config_mapping(config)
    configurable = mapping.get("configurable", {}) or {}
    metadata = mapping.get("metadata", {}) or {}
    auth_user = configurable.get("langgraph_auth_user", {}) or {}

    candidates = []
    if isinstance(auth_user, dict):
        candidates.append(auth_user.get("identity"))
    candidates.extend(
        (
            configurable.get("owner"),
            configurable.get("thread_owner"),
            metadata.get("owner"),
        )
    )
    identities = {str(value).strip() for value in candidates if value and str(value).strip()}
    if "default" in identities:
        raise MissingUserContextError("The reserved 'default' user is not a valid identity")
    if not identities:
        raise MissingUserContextError("Authenticated user context is required")
    if len(identities) != 1:
        raise UserContextMismatchError(
            f"Conflicting user identities in request context: {sorted(identities)}"
        )
    return identities.pop()


def require_runtime_user_id(runtime) -> str:
    """Resolve the authenticated user from a LangChain ToolRuntime."""
    return require_user_id(getattr(runtime, "config", None))


def require_thread_id(config) -> str:
    mapping = _config_mapping(config)
    thread_id = str((mapping.get("configurable", {}) or {}).get("thread_id") or "").strip()
    if not thread_id or thread_id == "default_thread":
        raise MissingThreadContextError("A concrete thread_id is required")
    return thread_id


def require_config_value(name: str, value: str | None) -> str:
    result = str(value or "").strip()
    if not result:
        raise RuntimeError(f"Required configuration {name} is not set")
    return result


def require_explicit_user_id(user_id: str | None) -> str:
    result = str(user_id or "").strip()
    if not result or result == "default":
        raise MissingUserContextError("A concrete user_id is required")
    return result

def mint_service_jwt(user_id: str) -> str:
    """Mint a short-lived JWT for the voice worker to call LangGraph Server.
    
    Uses the same JWT_SECRET as the main auth system so the LG auth handler
    can validate it. The 'sub' claim is the actual user_id so thread
    ownership is correctly attributed.
    """
    user_id = str(user_id or "").strip()
    if not user_id or user_id == "default":
        raise MissingUserContextError("A concrete user_id is required for a service JWT")
    secret = require_config_value("JWT_SECRET", config.JWT_SECRET)
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": "voice-worker",
        "service": True,
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, secret, algorithm=config.JWT_ALGORITHM)
