"""Authenticated access to the current user's generated images."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from routers.auth import get_current_user
from services.generated_images import image_path

router = APIRouter(prefix="/api/generated-images", tags=["generated-images"])


@router.get("/{image_id}")
async def get_generated_image(image_id: str, user: dict = Depends(get_current_user)):
    try:
        path = image_path(user["id"], image_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Image not found") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png", headers={
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
    })
