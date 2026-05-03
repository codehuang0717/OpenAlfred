from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from core.config import config

def localize_to_utc(time_str: str) -> str:
    """
    Normalize any time string into a canonical UTC ISO-8601 string with 'Z' suffix.

    Handles three input formats:
      1. Naive (no tz info, e.g. '2026-04-24T15:00:00')
         → Interpreted as the user's local timezone (config.TIMEZONE, e.g. Europe/London)
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
        if clean.endswith('Z'):
            # Already marked as UTC — parse it properly
            dt = datetime.fromisoformat(clean.replace('Z', '+00:00'))
        elif '+' in clean[10:] or (clean.count('-') > 2 and 'T' in clean):
            # Contains an explicit offset like +01:00 or -05:00
            dt = datetime.fromisoformat(clean)
        else:
            # Naive string — interpret as user's configured local timezone
            normalized = clean.replace(' ', 'T') if (' ' in clean and 'T' not in clean) else clean
            dt_naive = datetime.fromisoformat(normalized)
            user_tz = ZoneInfo(config.TIMEZONE)
            dt = dt_naive.replace(tzinfo=user_tz)

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
        dt = datetime.fromisoformat(clean.replace('Z', '+00:00'))
    else:
        dt = datetime.fromisoformat(clean)
    
    # If naive (no tzinfo), assume UTC (legacy DB entries)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(timezone.utc)


def utc_to_local(utc_str: str) -> str:
    """Convert a DB timestamp to a human-readable local time string.
    
    Input:  '2026-05-26T08:00:00Z' or '2026-03-11T09:00:00' (naive)
    Output: '2026-05-26 09:00 (BST)'  (if Europe/London in summer)
    
    Returns the original string if parsing fails.
    """
    if not utc_str:
        return ""
    try:
        dt = parse_to_aware_utc(utc_str)
        user_tz = ZoneInfo(config.TIMEZONE)
        local_dt = dt.astimezone(user_tz)
        tz_abbr = local_dt.strftime('%Z')  # e.g. 'BST', 'GMT'
        return local_dt.strftime(f"%Y-%m-%d %H:%M ({tz_abbr})")
    except Exception:
        return utc_str
