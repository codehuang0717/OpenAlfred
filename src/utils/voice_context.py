"""Strict parsing helpers for voice-session ownership metadata."""

from utils.auth_utils import require_explicit_user_id


def extract_outbound_user_id(room_name: str) -> str:
    """Extract a required user_id from a room name created by dial_user()."""
    if room_name.startswith("outbound-reminder-"):
        payload = room_name.removeprefix("outbound-reminder-")
        if len(payload) <= 37 or payload[36] != "-":
            raise ValueError(f"Malformed outbound reminder room name: {room_name}")
        return require_explicit_user_id(payload[37:])
    if room_name.startswith("outbound-supervisor-"):
        payload = room_name.removeprefix("outbound-supervisor-")
        _, separator, user_id = payload.partition("-")
        if not separator:
            raise ValueError(f"Malformed outbound supervisor room name: {room_name}")
        return require_explicit_user_id(user_id)
    if room_name.startswith("outbound-"):
        payload = room_name.removeprefix("outbound-")
        user_id, separator, timestamp = payload.rpartition("-")
        if not separator or not timestamp.isdigit():
            raise ValueError(f"Malformed outbound room name: {room_name}")
        return require_explicit_user_id(user_id)
    raise ValueError(f"Not an outbound room name: {room_name}")
