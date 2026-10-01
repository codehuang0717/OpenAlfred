import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.database import (
    get_email_credentials,
    set_email_credentials,
    delete_email_credentials,
)
from services.email import verify_account, EmailServiceException
from routers.auth import get_current_user

from schemas.responses import (
    EmailConfigResponse,
    EmailConfigUpdateResponse,
    EmailContentResponse,
    EmailResponse,
    StatusResponse,
    json_response,
)

router = APIRouter(prefix="/api", tags=["email"])
logger = logging.getLogger("email-router")

class EmailConfigRequest(BaseModel):
    account_id: Optional[str] = None
    email_address: str
    provider: str
    imap_server: str
    imap_port: int
    smtp_server: str
    smtp_port: int
    password: str

class EmailSendRequest(BaseModel):
    account_id: str
    to_address: str
    subject: str
    body: str

@router.get("/emails/recent", responses=json_response(list[EmailResponse], 200))
async def get_recent_emails_api(user: dict = Depends(get_current_user)):
    """Get recent emails from configured accounts."""
    from services.email import get_recent_emails, EmailServiceException
    try:
        emails = await get_recent_emails(user_id=user["id"], limit=15)
        return emails
    except EmailServiceException as e:
        return []
    except Exception as e:
        logger.error(f"Error fetching recent emails: {e}")
        return []

@router.get("/emails/{email_id}", responses=json_response(EmailContentResponse, 200))
async def get_email_api(email_id: str, account_id: str, user: dict = Depends(get_current_user)):
    """Get a specific email's content."""
    from services.email import read_email, EmailServiceException
    try:
        email_data = await read_email(user_id=user["id"], email_id=email_id, account_id=account_id)
        return email_data
    except EmailServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching email {email_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/emails/send", deprecated=True, status_code=410)
async def send_email_api(req: EmailSendRequest, user: dict = Depends(get_current_user)):
    """Old clients must import a persistent draft before confirming its version."""
    raise HTTPException(410, "请更新页面，通过已保存草稿确认发送；旧版直接发送接口已停用")

@router.get("/email/config", responses=json_response(list[EmailConfigResponse], 200))
async def get_email_configs_api(user: dict = Depends(get_current_user)):
    """Get the current user's email configurations."""
    creds = await get_email_credentials(user["id"])
    for cred in creds:
        if "encrypted_password" in cred:
            del cred["encrypted_password"]
    return creds

@router.post("/email/config", responses=json_response(EmailConfigUpdateResponse, 200))
async def set_email_config_api(req: EmailConfigRequest, user: dict = Depends(get_current_user)):
    """Add or update an email configuration after verifying."""
    from utils.crypto import encrypt_password
    
    try:
        await verify_account(
            imap_server=req.imap_server,
            imap_port=req.imap_port,
            smtp_server=req.smtp_server,
            smtp_port=req.smtp_port,
            email_address=req.email_address,
            password=req.password
        )
    except EmailServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    encrypted_pw = encrypt_password(req.password)
    account_id = req.account_id or str(uuid.uuid4())
    
    await set_email_credentials(
        account_id=account_id,
        user_id=user["id"],
        email_address=req.email_address,
        provider=req.provider,
        imap_server=req.imap_server,
        imap_port=req.imap_port,
        smtp_server=req.smtp_server,
        smtp_port=req.smtp_port,
        encrypted_password=encrypted_pw
    )
    
    return {"status": "success", "account_id": account_id}

@router.delete("/email/config/{account_id}", responses=json_response(StatusResponse, 200))
async def delete_email_config_api(account_id: str, user: dict = Depends(get_current_user)):
    """Delete an email configuration."""
    await delete_email_credentials(account_id=account_id, user_id=user["id"])
    return {"status": "deleted"}
