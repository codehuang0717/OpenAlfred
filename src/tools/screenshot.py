import base64
import io
import logging
from datetime import datetime, timezone
from services.tool_observations import observed, failed, fields
from PIL import ImageGrab
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from services.llm import get_model
from langchain.tools import ToolRuntime
from services.screen_monitor import require_screen_owner
from utils.auth_utils import require_runtime_user_id

logger = logging.getLogger("tools.screenshot")

@tool
async def take_screenshot(query: str, runtime: ToolRuntime) -> str:
    """Take a screenshot of the user's current screen and answer a specific query about it.
    Use this tool when the user asks you to look at their screen or asks what they are doing.
    Provide a specific question in the 'query' parameter to guide the visual analysis.
    """
    try:
        require_screen_owner(require_runtime_user_id(runtime))
        # Capture screen
        img = ImageGrab.grab()
        captured_at = datetime.now(timezone.utc).isoformat()
        buffered = io.BytesIO()
        # Convert to RGB to avoid alpha channel issues with JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large to save tokens/bandwidth
        max_size = (1920, 1080)
        img.thumbnail(max_size, ImageGrab.Image.Resampling.LANCZOS)
            
        img.save(buffered, format="JPEG", quality=70)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # We need a vision model. gpt-cloud usually supports vision.
        llm = get_model("gpt-cloud")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"This is a screenshot of the user's current screen. Please answer this query: {query}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                },
            ]
        )
        
        response = await llm.ainvoke([message])
        text = str(response.text)
        if not text:
            raise RuntimeError("Vision model returned no text answer")
        return observed(
            text,
            "已分析本次屏幕截图",
            details=fields(
                截取时间=captured_at,
                分析结果=text,
                说明="仅返回分析文字，没有保存可回放的截图",
            ),
        )
    except Exception as e:
        logger.error(f"Screenshot tool failed: {e}")
        return failed(
            f"Failed to capture or analyze screen: {str(e)}", "屏幕截取或分析失败"
        )


screenshot_tools = [take_screenshot]
