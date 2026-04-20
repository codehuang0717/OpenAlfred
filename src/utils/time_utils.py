from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import config

def localize_to_utc(time_str: str) -> str:
    """
    Standardizes a time string from the LLM into a UTC ISO-8601 string (ISO-Z).
    
    1. Strips any existing 'Z' (as LLMs often add it to local times by mistake).
    2. Interprets the naive time in the user's localized context (Europe/London).
    3. Converts it to UTC.
    4. Returns a standard ISO-Z string.
    
    Args:
        time_str: An ISO-like time string (e.g., '2026-04-16T15:00:00' or '2026-04-16T15:00:00Z')
        
    Returns:
        A UTC ISO-8601 string ending in 'Z'.
        
    Raises:
        ValueError: If the time_str cannot be parsed.
    """
    if not time_str:
        return ""
    
    # 幂等性检查：如果已经是 UTC 格式 (以 'Z' 结尾或包含 '+')，直接返回
    if time_str.endswith('Z') or '+' in time_str:
        return time_str
        
    try:
        # Strip 'Z' if present, because we want to mandate the local interpretation 
        # unless we explicitly decide to support multiple timezones in the future.
        naive_str = time_str.rstrip('Z')
        
        # Handle cases like "2026-04-16 10:00:00" vs "2026-04-16T10:00:00"
        if " " in naive_str and "T" not in naive_str:
            dt_naive = datetime.fromisoformat(naive_str.replace(" ", "T"))
        else:
            dt_naive = datetime.fromisoformat(naive_str)
            
        # Use config's centralized timezone (Europe/London)
        user_tz = ZoneInfo(config.TIMEZONE)
        
        # Localize the naive time to London (handling BST/GMT correctly)
        localized_dt = dt_naive.replace(tzinfo=user_tz)
        
        # Convert to UTC
        utc_dt = localized_dt.astimezone(timezone.utc)
        
        # Return standard ISO-8601 with Z suffix
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    except Exception as e:
        raise ValueError(f"Could not parse time string '{time_str}': {e}")
