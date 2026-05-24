import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Depends, Header, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from core.database import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    get_user_password_hash,
    update_user,
    update_user_last_login,
    update_user_password,
)
from core.config import config

logger = logging.getLogger("auth-router")


async def _regenerate_sip_config(db_path: str, output_path: str, ext: str):
    """Regenerate pjsip_users.conf and reload Asterisk.

    If CLOUD_HOST is set, SCPs the config to the cloud VM and reloads remotely.
    Otherwise reloads via local docker exec.

    Fire-and-forget — errors are logged but never raised to the caller.
    """
    import aiosqlite
    import os as _os
    import subprocess
    from datetime import datetime as _dt, timezone as _tz

    HEADER = ";; Dynamic user SIP endpoints — auto-generated\n;; Generated at: {ts}\n\n"
    ENDPOINT = (
        "[{ext}]\ntype=endpoint\ncontext=from-internal\ndisallow=all\n"
        "allow=ulaw,alaw\nauth=auth{ext}\naors={ext}\n"
        "transport=transport-udp\nrtp_symmetric=yes\nforce_rport=yes\n"
        "rewrite_contact=yes\ndirect_media=no\ntimers=no\n\n"
        "[auth{ext}]\ntype=auth\nauth_type=userpass\nusername={ext}\npassword={pwd}\n\n"
        "[{ext}]\ntype=aor\nmax_contacts=1\n\n"
    )

    try:
        # 1. Query users with SIP credentials
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT sip_extension, sip_password, username FROM users "
                "WHERE sip_extension != '' AND sip_extension IS NOT NULL "
                "AND CAST(sip_extension AS INTEGER) > 100 "
                "ORDER BY CAST(sip_extension AS INTEGER)"
            )
            users = await cursor.fetchall()

        # 2. Write config file locally
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(HEADER.format(ts=_dt.now(_tz.utc).isoformat()))
            if users:
                for u in users:
                    f.write(ENDPOINT.format(ext=u["sip_extension"], pwd=u["sip_password"]))
            else:
                f.write("; No users with SIP credentials yet.\n")

        logger.info(f"[_regenerate_sip_config] wrote {len(users)} endpoints to {output_path}")

        # 3. Push to cloud and reload Asterisk
        cloud_host = _os.getenv("CLOUD_HOST", "")
        cloud_dir = _os.getenv("CLOUD_DEPLOY_DIR", "~/cloud_deploy")

        if cloud_host:
            # ── Remote deployment: scp config + docker exec via ssh ──
            remote_conf = f"{cloud_dir}/asterisk/pjsip_users.conf"
            scp_result = subprocess.run(
                ["scp", output_path, f"{cloud_host}:{remote_conf}"],
                capture_output=True, text=True, timeout=15,
            )
            if scp_result.returncode == 0:
                logger.info(f"[_regenerate_sip_config] pushed config to {cloud_host}")
            else:
                logger.warning(f"[_regenerate_sip_config] scp failed: {scp_result.stderr.strip()}")
                return

            reload_result = subprocess.run(
                ["ssh", cloud_host,
                 f"cd {cloud_dir} && docker compose exec -T asterisk asterisk -rx 'pjsip reload'"],
                capture_output=True, text=True, timeout=15,
            )
            if reload_result.returncode == 0:
                logger.info(f"[_regenerate_sip_config] remote Asterisk PJSIP reloaded OK")
            else:
                logger.warning(f"[_regenerate_sip_config] remote reload failed: {reload_result.stderr.strip()}")
        else:
            # ── Local deployment: docker exec directly ──
            try:
                result = subprocess.run(
                    ["docker", "exec", "asterisk", "asterisk", "-rx", "pjsip reload"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.info(f"[_regenerate_sip_config] local Asterisk PJSIP reloaded OK")
                else:
                    logger.warning(f"[_regenerate_sip_config] reload rc={result.returncode}: {result.stderr.strip()}")
            except FileNotFoundError:
                logger.warning("[_regenerate_sip_config] docker CLI not found, skip reload")
            except Exception as e:
                logger.warning(f"[_regenerate_sip_config] docker exec failed: {e}")

    except Exception as e:
        logger.error(f"[_regenerate_sip_config] failed: {e}")
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
    """Register a new user account. Auto-assigns SIP extension and password."""
    logger.info(f"[register] username={req.username} display_name={req.display_name}")
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

    logger.info(
        f"[register] OK username={req.username} id={user_id} "
        f"sip_ext={user.get('sip_extension')}"
    )

    # ── Push SIP config to Asterisk (non-blocking fire-and-forget) ──
    try:
        import asyncio as _asyncio
        # cloud_deploy/ is inside agent/ (PROJECT_ROOT)
        cloud_deploy_dir = config.PROJECT_ROOT / "cloud_deploy"
        pjsip_users_conf = cloud_deploy_dir / "asterisk" / "pjsip_users.conf"

        # Run the generator inline (same process, reads same DB)
        _asyncio.create_task(_regenerate_sip_config(
            db_path=str(config.DB_PATH),
            output_path=str(pjsip_users_conf),
            ext=user.get("sip_extension", "?"),
        ))
    except Exception as e:
        logger.warning(f"[register] failed to trigger SIP config regeneration: {e}")

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
        "sip": {
            "extension": user.get("sip_extension"),
            "password": user.get("sip_password"),
            "server": config.LIVEKIT_URL.replace("ws://", "").replace("wss://", "").split(":")[0] if config.LIVEKIT_URL else "",
            "note": "使用此 SIP 账号密码登录软电话即可拨打 Alfred",
        },
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate and receive a JWT token."""
    logger.info(f"[login] username={req.username}")
    user = await get_user_by_username(req.username)
    if not user:
        logger.warning(f"[login] FAIL: username={req.username} not found")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        logger.warning(f"[login] FAIL: username={req.username} wrong password")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    await update_user_last_login(user["id"])
    token = create_jwt_token(user["id"], user["username"])

    logger.info(
        f"[login] OK username={req.username} id={user['id']}"
    )

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
        "sip": {
            "extension": user.get("sip_extension"),
            "password": user.get("sip_password"),
        } if user.get("sip_extension") else None,
    }


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Retrieve the profile of the currently authenticated user."""
    logger.debug(f"[me] user={user.get('username')} id={user.get('id')}")
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "created_at": user.get("created_at"),
        "sip_extension": user.get("sip_extension"),
    }


class UpdateMeRequest(BaseModel):
    display_name: str

@router.put("/me")
async def update_me(req: UpdateMeRequest, user: dict = Depends(get_current_user)):
    """Update the current user's profile (display_name)."""
    if not req.display_name or len(req.display_name.strip()) == 0:
        raise HTTPException(status_code=400, detail="Display name cannot be empty")
    await update_user(user["id"], display_name=req.display_name.strip())
    logger.info(f"[update_me] id={user['id']} display_name={req.display_name}")
    return {"status": "updated", "display_name": req.display_name.strip()}


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@router.put("/me/password")
async def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """Change the current user's password."""
    if not req.old_password or not req.new_password:
        raise HTTPException(status_code=400, detail="Both old and new passwords are required")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if req.new_password == req.old_password:
        raise HTTPException(status_code=400, detail="新密码不能与旧密码相同")

    stored_hash = await get_user_password_hash(user["id"])
    if not stored_hash or not bcrypt.checkpw(req.old_password.encode(), stored_hash.encode()):
        raise HTTPException(status_code=400, detail="旧密码不正确")

    new_hash = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
    await update_user_password(user["id"], new_hash)

    logger.info(f"[change_password] id={user['id']} password changed")
    return {"status": "updated"}


@router.post("/refresh")
async def refresh_token(user: dict = Depends(get_current_user)):
    """Issue a fresh JWT token for the authenticated user."""
    token = create_jwt_token(user["id"], user["username"])
    return {"token": token}
