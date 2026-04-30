import base64
import io
import logging
from PIL import ImageGrab
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llm import get_model

logger = logging.getLogger("tools.screenshot")

@tool
async def take_screenshot(query: str) -> str:
    """Take a screenshot of the user's current screen and answer a specific query about it.
    Use this tool when the user asks you to look at their screen or asks what they are doing.
    Provide a specific question in the 'query' parameter to guide the visual analysis.
    """
    try:
        # Capture screen
        img = ImageGrab.grab()
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
        return response.content
    except Exception as e:
        logger.error(f"Screenshot tool failed: {e}")
        return f"Failed to capture or analyze screen: {str(e)}"

screenshot_tools = [take_screenshot]
