import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Depends, Header, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from database import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    update_user_last_login,
)
from config import config

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# --- Pydantic Models ---

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = ""

class LoginRequest(BaseModel):
    username: str
    password: str

# --- JWT Helpers ---

def create_jwt_token(user_id: str, username: str) -> str:
    """Create a signed JWT token."""
    payload = {
        "sub": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=config.JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None)
) -> dict:
    """FastAPI dependency: extract and verify the current user from JWT."""
    jwt_token = None
    
    if credentials:
        jwt_token = credentials.credentials
    elif token:
        jwt_token = token
        
    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    payload = verify_jwt_token(jwt_token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Endpoints ---

@router.post("/register")
async def register(req: RegisterRequest):
    """Register a new user account."""
    if len(req.username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = await get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user_id = str(uuid.uuid4())
    password_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    display_name = req.display_name or req.username

    user = await create_user(
        id=user_id,
        username=req.username,
        display_name=display_name,
        password_hash=password_hash,
    )

    token = create_jwt_token(user_id, req.username)

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate and receive a JWT token."""
    user = await get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    await update_user_last_login(user["id"])
    token = create_jwt_token(user["id"], user["username"])

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
    }


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Retrieve the profile of the currently authenticated user."""
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "created_at": user.get("created_at"),
    }


@router.post("/refresh")
async def refresh_token(user: dict = Depends(get_current_user)):
    """Issue a fresh JWT token for the authenticated user."""
    token = create_jwt_token(user["id"], user["username"])
    return {"token": token}
