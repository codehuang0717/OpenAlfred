from __future__ import annotations

from typing import Optional

from langchain.tools import tool

from core.database import get_active_user
from services.weather import format_weather_text, get_weather_summary


async def _get_active_user_id() -> str | None:
    try:
        active_user = await get_active_user()
        if active_user:
            return active_user["id"]
    except Exception:
        pass
    return None


@tool
async def get_weather(
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    date: Optional[str] = None,
) -> str:
    """Get current weather and a short forecast for a city or place.

    Use this when the user asks about weather, temperature, rain, snow, wind,
    air conditions for going outside, or what to wear. If the user does not
    provide a location, call this tool without location so it can use the active
    user's saved default weather location. You may also pass latitude and
    longitude directly. The optional date may be a local date such as
    '2026-06-08' or a natural hint such as 'tomorrow'; when omitted, return
    current conditions plus the next 24 hours.
    """
    try:
        summary = await get_weather_summary(
            user_id=await _get_active_user_id(),
            location=location,
            latitude=latitude,
            longitude=longitude,
        )
    except Exception as exc:
        return f"Weather lookup failed: {exc}"
    return format_weather_text(summary, date=date)


weather_tools = [get_weather]
