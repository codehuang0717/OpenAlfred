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
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
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
    CLOUD_CHAT_MODEL = os.getenv("CLOUD_CHAT_MODEL", "gpt-5.4-nano")
    CLOUD_BROWSER_MODEL = os.getenv("CLOUD_BROWSER_MODEL", "gpt-5.4-mini")
    BARK_URL= os.getenv("BARK_URL", "https://api.day.app/BfQGU76aAZb9rJdWs2tNJW")

    # TTS Settings (Faster-Qwen3-TTS)
    TTS_URL = os.getenv("TTS_URL", "http://localhost:7017/v1/audio/speech")
    TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
    TTS_VOICE = os.getenv("TTS_VOICE", "yingxue")
    TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    TTS_JITTER_BUFFER_MS = int(os.getenv("TTS_JITTER_BUFFER_MS", "500"))

    # JWT Authentication Settings
    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # LangGraph Server
    LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://localhost:2024")

    # Email Settings
    EMAIL_ENCRYPTION_KEY = os.getenv("EMAIL_ENCRYPTION_KEY")

    # Context Management
    MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
    SUMMARY_THRESHOLD = int(os.getenv("SUMMARY_THRESHOLD", "15"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

    # Timezone Settings
    TIMEZONE = os.getenv("TIMEZONE", "Europe/London")

    # Browser Control Settings
    CHROME_CDP_URL = os.getenv("CHROME_CDP_URL", "http://localhost:9222")


    # Supervisor Settings
    SUPERVISOR_INTERVAL = int(os.getenv("SUPERVISOR_INTERVAL", "600")) # 10 minutes
    SUPERVISOR_PHONE_NUMBER = os.getenv("SUPERVISOR_PHONE_NUMBER", "100")
    SUPERVISOR_OCR_WINDOW_MINS = int(os.getenv("SUPERVISOR_OCR_WINDOW_MINS", "10"))

    # Redis Settings (Event Bus)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

config = Config()
