from __future__ import annotations

from typing import Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from services.tool_observations import observed, failed, fields, action

from langchain.tools import ToolRuntime, tool

from services.weather import format_weather_text, get_weather_summary
from utils.auth_utils import require_runtime_user_id


@tool
async def get_weather(
    runtime: ToolRuntime,
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
            user_id=require_runtime_user_id(runtime),
            location=location,
            latitude=latitude,
            longitude=longitude,
        )
    except Exception as exc:
        return failed(f"Weather lookup failed: {exc}", "天气查询失败")
    if not summary:
        return observed(
            format_weather_text(summary, date=date),
            "缺少查询地点，请设置位置或指定城市",
            outcome="blocked",
            actions=[action("settings", "位置设置", "profile")],
        )
    normalized = date
    if date in {"tomorrow", "明天", "today", "今天"}:
        today = datetime.now(ZoneInfo(summary["timezone"])).date()
        normalized = (
            today + timedelta(days=1 if date in {"tomorrow", "明天"} else 0)
        ).isoformat()
    days = [
        day
        for day in summary.get("daily_forecast", [])
        if not normalized or day.get("date") == normalized
    ]
    current = summary.get("current", {})
    text = format_weather_text(summary, date=normalized)
    if normalized and not days:
        text += "\n请求日期不在已返回的预报范围内。"
    return observed(
        text,
        f"{summary['location'].get('label')} · "
        + (
            f"{normalized} 预报"
            if normalized and days
            else "请求日期无预报"
            if normalized
            else f"{current.get('weather')}，{current.get('temperature')}°C"
        )
        + ("（过期缓存）" if summary.get("stale") else ""),
        outcome="partial"
        if summary.get("stale") or normalized and not days
        else "completed",
        details=fields(
            实际地点=summary["location"].get("label"),
            当地时区=summary.get("timezone"),
            观测时间=current.get("time"),
            更新于=summary.get("updated_at"),
            数据状态="过期缓存" if summary.get("stale") else "本次可用数据",
            当前天气=f"{current.get('weather')} · {current.get('temperature')}°C · 风速 {current.get('wind_speed')} km/h",
            预报="\n".join(
                f"{day['date']} · {day.get('weather')} · {day.get('temperature_min')}–{day.get('temperature_max')}°C · 降雨概率 {day.get('rain_probability')}%"
                for day in days
            ),
        ),
        actions=[action("external", "天气数据来源", "https://open-meteo.com/")],
    )


weather_tools = [get_weather]
