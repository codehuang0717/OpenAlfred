# src/config.py
import os
import dotenv
from pathlib import Path

# Load environment variables
# Find project root (assuming src/core is inside agent folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
ENV_PATH = PROJECT_ROOT / ".env"

dotenv.load_dotenv(ENV_PATH, override=True)

class Config:
    """Central configuration for OpenAlfred."""

    # Paths
    PROJECT_ROOT = PROJECT_ROOT
    ASSETS_DIR = PROJECT_ROOT / "assets"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Memory System (L1: local .md files)
    MEMORY_DIR = PROJECT_ROOT.parent / "memory"  # repo root / memory/
    EXTRACTION_INTERVAL = int(os.getenv("EXTRACTION_INTERVAL", "3"))
    EXTRACTION_MIN_MSG_LENGTH = int(os.getenv("EXTRACTION_MIN_MSG_LENGTH", "8"))
    
    # LiveKit Settings
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID", "ST_Bcj2LDXqL4J7")
    
    # Database Settings
    DB_PATH = PROJECT_ROOT / "todos.db"

    # Multi-Provider API Keys
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    # Model Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "gemma4:e2b")
    CLOUD_CHAT_MODEL = os.getenv("CLOUD_CHAT_MODEL", "gpt-5.4-nano")
    CLOUD_BROWSER_MODEL = os.getenv("CLOUD_BROWSER_MODEL", "gpt-5.4-mini")
    CEREBRAS_CHAT_MODEL = os.getenv("CEREBRAS_CHAT_MODEL", "llama-4-scout")
    GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_FLASH_MODEL = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")
    DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
    BARK_URL= os.getenv("BARK_URL", "https://api.day.app/BfQGU76aAZb9rJdWs2tNJW")

    # TTS Settings (Faster-Qwen3-TTS)
    TTS_URL = os.getenv("TTS_URL", "http://localhost:7017/v1/audio/speech")
    TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
    TTS_VOICE = os.getenv("TTS_VOICE", "yingxue")
    TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    TTS_JITTER_BUFFER_MS = int(os.getenv("TTS_JITTER_BUFFER_MS", "80"))

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
    CHROME_USER_DATA_DIR = os.getenv(
        "CHROME_USER_DATA_DIR",
        str(PROJECT_ROOT / "agent_chrome_profile")
    )

    # SenseVoice STT Settings
    SENSEVOICE_STT_URL = os.getenv("SENSEVOICE_STT_URL", "http://127.0.0.1:8000/extract_text")

    # Screenpipe Settings (eye tool)
    SCREENPIPE_URL = os.getenv("SCREENPIPE_URL", "http://localhost:3030")

    # Greeting TTS Settings
    GREETING_TTS_URL = os.getenv("GREETING_TTS_URL", "http://127.0.0.1:10096/tts/stream")

    # Local LiveKit WS (for ear service)
    LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL", "ws://localhost:7880")

    # Supervisor Settings
    SUPERVISOR_INTERVAL = int(os.getenv("SUPERVISOR_INTERVAL", "600")) # 10 minutes
    SUPERVISOR_PHONE_NUMBER = os.getenv("SUPERVISOR_PHONE_NUMBER", "100")
    SUPERVISOR_OCR_WINDOW_MINS = int(os.getenv("SUPERVISOR_OCR_WINDOW_MINS", "10"))

    # MCP Client Settings (connect to external MCP servers)
    # JSON string: {"server_name": {"command": "npx", "args": [...], "transport": "stdio"}}
    # or for HTTP: {"server_name": {"url": "http://...", "transport": "http"}}
    MCP_SERVERS_CONFIG = os.getenv("MCP_SERVERS_CONFIG", "")

    # MCP Server Settings (expose OpenAlfred tools to other AI apps)
    MCP_SERVER_ENABLED = os.getenv("MCP_SERVER_ENABLED", "false").lower() == "true"
    MCP_SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "sse")
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8100"))
    MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "OpenAlfred")

    # Redis Settings (Event Bus)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

config = Config()
