from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from browser_use import Agent, Browser, ChatOpenAI
from services.llm import get_model
from core.config import config
import logging
import asyncio

import socket
import subprocess
import os
import time

logger = logging.getLogger("browser-tool")

def get_chrome_path():
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Users\12611\AppData\Local\Google\Chrome\Application\chrome.exe"
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            return path
    return None

@tool
async def web_browser_task(runtime: ToolRuntime, task_description: str) -> Command:
    """Browse the web, interact with websites, or extract info. Provide a detailed task_description."""
    logger.info(f"Starting browser task: {task_description}")
    
    # 1. Get the LLM using browser-use's specific wrapper to avoid Pydantic issues
    llm = ChatOpenAI(model=config.CLOUD_BROWSER_MODEL, api_key=config.OPENAI_API_KEY)

    # 2. Find local Chrome path
    chrome_path = get_chrome_path()
    if not chrome_path:
        logger.error("Could not find Google Chrome on this system.")
        return Command(update={"messages": [ToolMessage("Failed: Chrome not found.", tool_call_id=runtime.tool_call_id)]})

    # 3. Launch Chrome manually using subprocess.Popen
    # This avoids the 'NotImplementedError' from asyncio.create_subprocess_exec on some Windows event loops
    profile_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "agent_chrome_profile")
    os.makedirs(profile_dir, exist_ok=True)
    
    port = 9222
    logger.info(f"Launching independent Chrome instance on port {port} with profile {profile_dir}")
    
    # We use subprocess.Popen because it's synchronous and doesn't rely on the asyncio loop's support for subprocesses
    subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check"
    ])
    
    # Wait for Chrome to initialize the CDP server
    await asyncio.sleep(3)
    
    try:
        # 4. Connect browser-use to the manually launched instance via CDP
        browser = Browser(cdp_url=f"http://127.0.0.1:{port}")
        
        # 5. Initialize the browser-use sub-agent
        agent = Agent(
            task=task_description,
            llm=llm,
            browser=browser,
        )
        
        # 4. Execute the browser task
        result = await agent.run()
        
        final_result = result.final_result()
        logger.info(f"Browser task completed. Result: {final_result}")
        
    except Exception as e:
        logger.error(f"Browser task failed: {str(e)}")
        final_result = f"Failed to execute browser task. Error: {str(e)}. Make sure Chrome is running with --remote-debugging-port=9222."

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=final_result,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

browser_tools = [web_browser_task]
