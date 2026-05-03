import time
import jwt as pyjwt
from core.config import config

def mint_service_jwt(user_id: str) -> str:
    """Mint a short-lived JWT for the voice worker to call LangGraph Server.
    
    Uses the same JWT_SECRET as the main auth system so the LG auth handler
    can validate it. The 'sub' claim is the actual user_id so thread
    ownership is correctly attributed.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": "voice-worker",
        "service": True,
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
