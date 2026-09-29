from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo
from services.user_time import validate_timezone

_END_OF_DAY = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ]24:00(?::00(?:[.,]0+)?)?(Z|[+-]\d{2}:\d{2})?$"
)


def _parse_iso_datetime(value: str) -> datetime:
    """Normalize the exact end-of-day notation; reject other invalid hours."""
    match = _END_OF_DAY.fullmatch(value)
    if match:
        day, offset = match.groups()
        midnight = f"{day}T00:00:00{offset or ''}".replace("Z", "+00:00")
        return datetime.fromisoformat(midnight) + timedelta(days=1)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def localize_to_utc(time_str: str, timezone_name: str | None = None) -> str:
    """
    Normalize any time string into a canonical UTC ISO-8601 string with 'Z' suffix.

    Handles three input formats:
      1. Naive (no tz info, e.g. '2026-04-24T15:00:00')
         → Requires an explicit IANA timezone supplied by the caller.
         → Converted to UTC.
      2. 'Z'-suffixed (e.g. '2026-04-24T14:00:00Z')
         → Parsed as UTC, re-formatted for consistency.
      3. Offset-aware (e.g. '2026-04-24T15:00:00+01:00')
         → Parsed with the given offset, converted to UTC.

    This function is safe to call multiple times on the same value (idempotent).

    Returns:
        A UTC ISO-8601 string ending in 'Z', e.g. '2026-04-24T14:00:00Z'.

    Raises:
        ValueError: If the time_str cannot be parsed.
    """
    if not time_str:
        return ""

    clean = time_str.strip()

    try:
        dt = _parse_iso_datetime(clean)
        if dt.tzinfo is None:
            zone = ZoneInfo(validate_timezone(timezone_name))
            candidates = []
            for fold in (0, 1):
                candidate = dt.replace(tzinfo=zone, fold=fold)
                if candidate.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == dt:
                    candidates.append(candidate)
            if not candidates:
                raise ValueError("此本地时间因夏令时切换不存在，请选择其他时间")
            if len({c.utcoffset() for c in candidates}) > 1:
                raise ValueError("此本地时间因夏令时切换出现两次，请提供明确的 UTC 偏移量")
            dt = candidates[0]

        # Convert to UTC and return canonical format
        utc_dt = dt.astimezone(timezone.utc)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    except Exception as e:
        raise ValueError(f"Could not parse time string '{time_str}': {e}")


def parse_to_aware_utc(time_str: str) -> datetime:
    """Parse any DB timestamp into a timezone-aware UTC datetime.
    
    Handles all formats found in the DB:
      - '2026-05-26T08:00:00Z'         → aware UTC
      - '2026-05-26T08:00:00+00:00'    → aware UTC
      - '2026-03-11T09:00:00'          → naive, treated as UTC
      - '2026-02-27T20:00:00+08:00'    → offset-aware, converted to UTC
    
    Raises ValueError if unparseable.
    """
    if not time_str:
        raise ValueError("Empty time string")
    
    clean = time_str.strip()
    
    if clean.endswith('Z'):
        dt = _parse_iso_datetime(clean)
    else:
        dt = _parse_iso_datetime(clean)
    
    # If naive (no tzinfo), assume UTC (legacy DB entries)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(timezone.utc)


def utc_to_local(utc_str: str, timezone_name: str) -> str:
    """Convert a DB timestamp to a human-readable local time string.
    
    Input:  '2026-05-26T08:00:00Z' or '2026-03-11T09:00:00' (naive)
    Output includes the IANA timezone and current UTC offset.
    
    Raises ValueError if parsing fails.
    """
    if not utc_str:
        return ""
    dt = parse_to_aware_utc(utc_str)
    user_tz = ZoneInfo(validate_timezone(timezone_name))
    local_dt = dt.astimezone(user_tz)
    return f"{local_dt:%Y-%m-%d %H:%M} ({timezone_name}, UTC{local_dt:%z})"
