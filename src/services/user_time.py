"""User-scoped IANA timezones shared by chat, tools and background work."""

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any

from db.settings import get_setting, set_setting
from utils.auth_utils import require_explicit_user_id


class MissingTimezoneError(ValueError):
    pass


def runtime_timezone(runtime: Any) -> str:
    """Read the run's timezone snapshot; never re-read a changing profile."""
    conf = (runtime.config or {}).get("configurable", {})
    if "timezone" in conf:
        return validate_timezone(conf["timezone"])
    state = getattr(runtime, "state", None)
    value = state.get("user_timezone") if isinstance(state, dict) else getattr(state, "user_timezone", None)
    return validate_timezone(value)


def validate_timezone(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MissingTimezoneError("尚未取得用户时区，请从浏览器打开聊天后重试")
    value = value.strip()
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"无效的 IANA 时区：{value}") from exc
    return value


async def save_user_timezone(user_id: str, value: str) -> str:
    user_id = require_explicit_user_id(user_id)
    value = validate_timezone(value)
    key = f"user_timezone:v1:{user_id}"
    if await get_setting(key) != value:
        await set_setting(key, value)
    return value


async def get_user_timezone(user_id: str, run_config: dict | None = None) -> str:
    user_id = require_explicit_user_id(user_id)
    conf = (run_config or {}).get("configurable", {})
    if "timezone" in conf:
        # The caller's current device timezone is fixed for this run, even if
        # another session later updates the saved timezone for background work.
        return validate_timezone(conf["timezone"])
    return validate_timezone(await get_setting(f"user_timezone:v1:{user_id}"))
