import logging
import httpx
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from routers.auth import get_current_user
from core.config import config
from services.voice_store import get_voice_settings

from schemas.responses import (
    TextResponse,
    json_response,
)

router = APIRouter(prefix="/api", tags=["multimodal"])
logger = logging.getLogger("multimodal-router")

@router.post("/transcribe", responses=json_response(TextResponse, 200))
async def transcribe_audio_api(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Transcribe audio using local SenseVoice API."""
    if not (await get_voice_settings(user["id"])).stt_enabled:
        raise HTTPException(status_code=409, detail="语音识别已关闭，请在语音设置中开启 STT")
    try:
        content = await file.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "录音不能超过 20MB")
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, content, file.content_type)}
            resp = await client.post(
                config.SENSEVOICE_STT_URL, files=files, timeout=30.0
            )
            if resp.status_code == 200:
                result = resp.json()
                return {"text": result.get("results", "")}
            else:
                raise HTTPException(status_code=resp.status_code, detail="Transcription failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parse-file", responses=json_response(TextResponse, 200))
async def parse_file_api(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Extract text from uploaded files (PDF, TXT, etc.)."""
    try:
        content = await file.read()
        text = ""
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".docx"):
            import io
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = content.decode('utf-8', errors='ignore')
            
        return {"text": text.strip()}
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")
