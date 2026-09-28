"""
LangGraph Auth Handler — JWT-based authentication and thread-level authorization.

Integrates with LangGraph Server's built-in auth system to:
1. Validate JWT tokens on every request
2. Bind threads to the authenticated user via metadata
3. Enforce user isolation: users can only access their own threads
"""

import jwt
from utils.logger import get_logger
from langgraph_sdk import Auth
from core.config import config
from utils.auth_utils import MissingUserContextError, require_config_value, require_explicit_user_id

logger = get_logger("auth")

auth = Auth()


@auth.authenticate
async def authenticate_user(
    authorization: str | None,
) -> Auth.types.MinimalUserDict:
    """Validate the JWT token from the Authorization header.

    Supports two modes:
    1. Normal user JWT (from web frontend)
    2. Service-account JWT (from voice worker) — identified by "service": True in payload.
       The voice worker mints these tokens using the shared JWT_SECRET and passes
       the actual user_id as the "sub" claim so thread ownership is correctly assigned.

    Returns the user's identity, which LangGraph injects into the request context
    and makes available via config["configurable"]["langgraph_auth_user"].
    """
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing authorization header")

    # Extract Bearer token
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid authorization scheme")

    try:
        secret = require_config_value("JWT_SECRET", config.JWT_SECRET)
        payload = jwt.decode(token, secret, algorithms=[config.JWT_ALGORITHM])
        user_id = payload.get("sub")
        username = payload.get("username", "")
        try:
            user_id = require_explicit_user_id(user_id)
        except MissingUserContextError:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token payload")
    except jwt.ExpiredSignatureError:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    return {
        "identity": user_id,
        "display_name": username,
        "is_authenticated": True,
    }


# ─── Resource-Level Access Control ─────────────────────────────────────────


@auth.on.threads.create
async def on_thread_create(ctx: Auth.types.AuthContext, value: dict):
    """Automatically tag new threads with the creator's user ID."""
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}


@auth.on.threads.read
async def filter_read_threads(ctx: Auth.types.AuthContext, value: dict):
    """Users can only read threads they own."""
    return {"owner": ctx.user.identity}


@auth.on.threads.update
async def filter_update_threads(ctx: Auth.types.AuthContext, value: dict):
    """Users can only update threads they own."""
    return {"owner": ctx.user.identity}


@auth.on.threads.delete
async def filter_delete_threads(ctx: Auth.types.AuthContext, value: dict):
    """Users can only delete threads they own."""
    return {"owner": ctx.user.identity}


@auth.on.threads.search
async def filter_search_threads(ctx: Auth.types.AuthContext, value: dict):
    """Thread search results are scoped to the authenticated user."""
    return {"owner": ctx.user.identity}
