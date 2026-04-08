# src/config.py
import os
import dotenv
from pathlib import Path

# Load environment variables
# Find project root (assuming src is inside agent folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ENV_PATH = PROJECT_ROOT / ".env"

dotenv.load_dotenv(ENV_PATH, override=True)

class Config:
    """Central configuration for OpenAlfred."""
    
    # LLM Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    
    # Memory Settings
    MEM0_API_KEY = os.getenv("MEM0_API_KEY")
    
    # LiveKit Settings
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    
    # Database Settings
    DB_PATH = PROJECT_ROOT / "todos.db"
    
    # Asset Settings
    ASSETS_DIR = PROJECT_ROOT / "assets"
    
    # Local Model Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "gemma4:e2b")
    CLOUD_MODEL_NAME = os.getenv("CLOUD_MODEL_NAME", "gpt-5.4-mini") # Use a real name by default
    BARK_URL= os.getenv("BARK_URL", "https://api.day.app/BfQGU76aAZb9rJdWs2tNJW")

config = Config()
