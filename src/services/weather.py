from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from core.database import get_setting, set_setting


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REVERSE_GEOCODING_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
WEATHER_LOCATION_KEY_PREFIX = "weather_location"
WEATHER_CACHE_TTL_SECONDS = 10 * 60

WEATHER_CODES = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}

_weather_cache: dict[str, tuple[float, dict]] = {}


def weather_location_key(user_id: str) -> str:
    return f"{WEATHER_LOCATION_KEY_PREFIX}:{user_id}"


async def get_saved_weather_location(user_id: str) -> dict | None:
    raw = await get_setting(weather_location_key(user_id), "")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if latitude is None or longitude is None:
        return None
    return data


async def save_weather_location(user_id: str, location: dict) -> dict:
    label = (location.get("label") or "当前位置").strip() or "当前位置"
    if label in {"当前位置", "已保存的位置"}:
        label = await reverse_geocode_label(location["latitude"], location["longitude"]) or label

    payload = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "accuracy": location.get("accuracy"),
        "label": label,
        "source": (location.get("source") or "browser").strip() or "browser",
    }
    await set_setting(weather_location_key(user_id), json.dumps(payload, ensure_ascii=False))
    return payload


async def reverse_geocode_label(latitude: float, longitude: float) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                REVERSE_GEOCODING_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "localityLanguage": "zh",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    city = data.get("city") or data.get("locality")
    principal = data.get("principalSubdivision")
    country = data.get("countryName")
    parts = []
    for part in (city, principal, country):
        if part and part not in parts:
            parts.append(part)
    return ", ".join(parts) if parts else None


async def clear_weather_location(user_id: str) -> None:
    await set_setting(weather_location_key(user_id), "")


def weather_text(code: Optional[int]) -> str:
    if code is None:
        return "未知"
    return WEATHER_CODES.get(code, f"未知天气代码 {code}")


def _format_place(result: dict) -> str:
    parts = [result.get("name")]
    admin = result.get("admin1")
    country = result.get("country")
    if admin and admin != parts[0]:
        parts.append(admin)
    if country:
        parts.append(country)
    return ", ".join(part for part in parts if part)


def _safe_index(items: list, index: int):
    if 0 <= index < len(items):
        return items[index]
    return None


def _nearest_hour_index(times: list[str], current_time: str | None) -> int:
    if not times:
        return 0
    if current_time in times:
        return times.index(current_time)

    if current_time:
        try:
            current_dt = datetime.fromisoformat(current_time)
            parsed = [datetime.fromisoformat(t) for t in times]
            return min(range(len(parsed)), key=lambda i: abs(parsed[i] - current_dt))
        except Exception:
            pass
    return 0


def _cache_key(latitude: float, longitude: float) -> str:
    return f"{float(latitude):.4f},{float(longitude):.4f}"


def _with_location(summary: dict, location: dict, *, stale: bool | None = None) -> dict:
    result = dict(summary)
    result["location"] = {
        "label": location.get("label") or "已保存的位置",
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "accuracy": location.get("accuracy"),
        "source": location.get("source"),
    }
    if stale is not None:
        result["stale"] = stale
    return result


async def resolve_weather_location(
    *,
    user_id: str | None = None,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict | None:
    query = location.strip() if location else ""
    if latitude is not None and longitude is not None:
        return {
            "latitude": latitude,
            "longitude": longitude,
            "label": query or "给定坐标",
            "source": "coordinates",
        }

    if query:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                GEOCODING_URL,
                params={
                    "name": query,
                    "count": 1,
                    "language": "zh",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            geo_data = resp.json()

        results = geo_data.get("results") or []
        if not results:
            return None
        place = results[0]
        return {
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "label": _format_place(place),
            "source": "geocoding",
        }

    if user_id:
        return await get_saved_weather_location(user_id)
    return None


def _build_summary(location: dict, forecast: dict, *, stale: bool = False) -> dict:
    current = forecast.get("current", {})
    hourly = forecast.get("hourly", {})
    daily = forecast.get("daily", {})
    current_time = current.get("time")
    hour_times = hourly.get("time", [])
    hour_index = _nearest_hour_index(hour_times, current_time)

    precip_probs = hourly.get("precipitation_probability", [])
    hourly_codes = hourly.get("weather_code", [])
    hourly_temps = hourly.get("temperature_2m", [])
    hourly_winds = hourly.get("wind_speed_10m", [])
    upcoming = []
    for offset in (3, 6, 12, 24):
        idx = hour_index + offset
        time_label = _safe_index(hour_times, idx)
        if not time_label:
            continue
        upcoming.append(
            {
                "time": time_label,
                "weather": weather_text(_safe_index(hourly_codes, idx)),
                "temperature": _safe_index(hourly_temps, idx),
                "rain_probability": _safe_index(precip_probs, idx),
                "wind_speed": _safe_index(hourly_winds, idx),
            }
        )

    daily_times = daily.get("time", [])
    daily_codes = daily.get("weather_code", [])
    daily_max = daily.get("temperature_2m_max", [])
    daily_min = daily.get("temperature_2m_min", [])
    daily_rain = daily.get("precipitation_probability_max", [])
    daily_items = []
    for idx, day in enumerate(daily_times[:3]):
        daily_items.append(
            {
                "date": day,
                "weather": weather_text(_safe_index(daily_codes, idx)),
                "temperature_min": _safe_index(daily_min, idx),
                "temperature_max": _safe_index(daily_max, idx),
                "rain_probability": _safe_index(daily_rain, idx),
            }
        )

    today = daily_items[0] if daily_items else {}
    current_weather = weather_text(current.get("weather_code"))
    rain_probability = today.get("rain_probability")
    wind_speed = current.get("wind_speed_10m")
    apparent = current.get("apparent_temperature")
    suggestions = []
    if rain_probability is not None and rain_probability >= 50:
        suggestions.append("今天降水概率偏高，出门记得带伞。")
    if apparent is not None and apparent <= 12:
        suggestions.append("体感温度偏低，外出建议多穿一层。")
    if wind_speed is not None and wind_speed >= 18:
        suggestions.append("风比较明显，户外活动注意保暖和防风。")
    if not suggestions:
        suggestions.append("天气整体比较平稳，可以按原计划安排外出。")
    suggestions.append("可以问 Alfred：今天适合跑步或通勤吗？")

    return {
        "location": {
            "label": location.get("label") or "已保存的位置",
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "accuracy": location.get("accuracy"),
            "source": location.get("source"),
        },
        "timezone": forecast.get("timezone", "unknown"),
        "current": {
            "time": current_time,
            "weather": current_weather,
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": apparent,
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": wind_speed,
        },
        "daily": today,
        "daily_forecast": daily_items,
        "upcoming": upcoming,
        "suggestions": suggestions,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stale": stale,
    }


async def get_weather_summary(
    *,
    user_id: str | None = None,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_cache: bool = True,
) -> dict | None:
    resolved = await resolve_weather_location(
        user_id=user_id,
        location=location,
        latitude=latitude,
        longitude=longitude,
    )
    if not resolved:
        return None
    if (resolved.get("label") or "") in {"当前位置", "已保存的位置"}:
        label = await reverse_geocode_label(resolved["latitude"], resolved["longitude"])
        if label:
            resolved = dict(resolved)
            resolved["label"] = label

    cache_key = _cache_key(resolved["latitude"], resolved["longitude"])
    now = time.monotonic()
    cached = _weather_cache.get(cache_key)
    if use_cache and cached and now - cached[0] < WEATHER_CACHE_TTL_SECONDS:
        return _with_location(cached[1], resolved)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": resolved["latitude"],
                    "longitude": resolved["longitude"],
                    "current": ",".join(
                        [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "precipitation",
                            "weather_code",
                            "wind_speed_10m",
                        ]
                    ),
                    "hourly": ",".join(
                        [
                            "temperature_2m",
                            "precipitation_probability",
                            "weather_code",
                            "wind_speed_10m",
                        ]
                    ),
                    "daily": ",".join(
                        [
                            "weather_code",
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                        ]
                    ),
                    "forecast_days": 3,
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            forecast = resp.json()
    except Exception:
        if cached:
            return _with_location(cached[1], resolved, stale=True)
        raise

    summary = _build_summary(resolved, forecast)
    _weather_cache[cache_key] = (now, summary)
    return summary


def format_weather_text(summary: dict, date: str | None = None) -> str:
    if not summary:
        return (
            "Weather lookup failed: no location was provided and no default "
            "weather location is saved. Please provide a city or enable current "
            "location in Settings."
        )

    location = summary["location"]
    current = summary["current"]
    lines = [
        f"地点: {location.get('label')}",
        f"坐标: {location.get('latitude')}, {location.get('longitude')}",
        f"当地时区: {summary.get('timezone', 'unknown')}",
        f"当前时间: {current.get('time')}",
        "当前天气: {weather}, 气温 {temp}°C, 体感 {feels}°C, 湿度 {humidity}%, 降水 {precip} mm, 风速 {wind} km/h".format(
            weather=current.get("weather"),
            temp=current.get("temperature"),
            feels=current.get("apparent_temperature"),
            humidity=current.get("humidity"),
            precip=current.get("precipitation"),
            wind=current.get("wind_speed"),
        ),
    ]

    daily_lines = []
    for idx, day in enumerate(summary.get("daily_forecast", [])):
        if date and date not in day.get("date", "") and idx > 0:
            continue
        daily_lines.append(
            "- {day}: {weather}, {low}-{high}°C, 最高降水概率 {rain}%".format(
                day=day.get("date"),
                weather=day.get("weather"),
                low=day.get("temperature_min"),
                high=day.get("temperature_max"),
                rain=day.get("rain_probability"),
            )
        )

    if daily_lines:
        lines.append("每日预报:")
        lines.extend(daily_lines)

    upcoming = summary.get("upcoming", [])
    if upcoming:
        lines.append("未来 24 小时要点:")
        for item in upcoming:
            lines.append(
                "- {time}: {weather}, {temp}°C, 降水概率 {rain}%, 风速 {wind} km/h".format(
                    time=str(item.get("time", "")).replace("T", " "),
                    weather=item.get("weather"),
                    temp=item.get("temperature"),
                    rain=item.get("rain_probability"),
                    wind=item.get("wind_speed"),
                )
            )

    return "\n".join(lines)


def format_weather_prompt_context(summary: dict | None) -> str:
    if not summary:
        return ""

    location = summary["location"]
    current = summary["current"]
    daily = summary.get("daily") or {}
    suggestions = "；".join(summary.get("suggestions", [])[:2])
    stale_note = " Yes" if summary.get("stale") else " No"
    return "\n".join(
        [
            "[天气信息]",
            f"Location: {location.get('label')} ({location.get('latitude')}, {location.get('longitude')})",
            f"Current: {current.get('weather')}, {current.get('temperature')}°C, feels like {current.get('apparent_temperature')}°C",
            f"Today: {daily.get('weather')}, {daily.get('temperature_min')}-{daily.get('temperature_max')}°C, rain probability {daily.get('rain_probability')}%",
            f"Wind: {current.get('wind_speed')} km/h. Humidity: {current.get('humidity')}%.",
            f"Suggestions: {suggestions}",
            f"Updated At: {summary.get('updated_at')}. Stale: {stale_note}",
        ]
    )
