<div align="center">
  <img src="https://raw.githubusercontent.com/codehuang0717/OpenAlfred/main/assets/logo.png" alt="OpenAlfred Logo" width="120" />
  <h1>OpenAlfred</h1>
  <p><b>A powerful, stateful, and modular AI Agent framework</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/LangGraph-Latest-orange?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/LiveKit-Ready-brightgreen?style=flat-square" alt="LiveKit" />
    <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License" />
    <img src="https://img.shields.io/badge/Version-v1.3.0-blue?style=flat-square" alt="Version" />

  </p>
</div>

---

## 🌟 Overview

OpenAlfred is an advanced AI agent backend built on top of **LangGraph** and **LiveKit**. It features a modern, service-oriented architecture designed for reliability, low latency, and deep modularity.

Whether it's managing your reminders, holding contextual long-term conversations, or handling proactive user monitoring, OpenAlfred is built to be a robust foundation for next-generation AI assistants.

## ✨ Key Features & Recent Updates

- 🎙️ **Voice Wakeup & Interruption**: Low-latency STT/TTS pipeline with VAD buffering, supporting seamless interruptions.
- 🧠 **Dual-Layer Supervisor**: A proactive background monitoring system that pushes notifications and initiates conversations independently.
- 📧 **Email Integration**: Built-in email processing and notification capabilities.
- 🧩 **Modular Architecture**: Decoupled Event-Driven design via Redis Pub/Sub for seamless inter-service communication.
- 💾 **Context & Memory**: Intelligent context management and long-term memory via SQLite and Mem0.

- 🎯 **Structured Output**: Multi-provider structured output utility with automatic native/JSON fallback — ensures LLM responses are type-safe Pydantic models across GPT, DeepSeek, Gemini, and Ollama.

## 🏗️ Deep Architecture

### 1. LangGraph Workflow
The core logic is powered by LangGraph state machines. The agent transitions seamlessly between evaluating the LLM's decisions, executing tools (like checking emails or fetching context), and returning final structured outputs.

### 2. LiveKit Pipeline
For voice interactions, the `livekit_service` maintains an efficient pipeline:
`User Audio -> VAD (Voice Activity Detection) -> STT -> LLM Processing -> TTS -> Room Audio`

### 3. Proactive Supervisor
Unlike standard reactive bots, OpenAlfred incorporates an independent Supervisor service that evaluates context out-of-band and pushes real-time event triggers to the active session when it detects the need for proactive engagement.

## 📂 Project Structure

```text
agent/src/
├── core/             # Configuration, Auth, and Event Bus setup
├── logic/            # LangGraph Nodes, Prompts, and Graph definitions
├── services/         # Business services (Email, TTS, LLM, Memory, etc.)
├── livekit_service/  # Voice interaction layer (Session management, STT, Playback)
├── db/               # Database access layer (SQLite/Mem0)
├── routers/          # FastAPI API endpoints
├── tools/            # Agent toolsets
└── utils/            # Shared utilities (Logging, Latency tracking)
```

## 🚀 Getting Started (Comprehensive Setup)

### Prerequisites

- Python 3.13+ & [uv](https://github.com/astral-sh/uv)
- Redis Server (local or remote)
- LiveKit Server (local or Cloud)
- LLM API keys (OpenAI / compatible)

### Installation & Run

1. **Install Prerequisites**:
   ```bash
   # Windows: Install Redis and LiveKit Server
   winget install Redis.Redis
   # Download LiveKit binary from https://github.com/livekit/livekit/releases
   # and place it in OpenAlfred/bin/
   ```

2. **Clone & Sync**:
   ```bash
   git clone https://github.com/codehuang0717/OpenAlfred.git
   cd OpenAlfred/agent
   uv sync
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Fill in your OPENAI_API_KEY, etc.
   # REDIS_URL and LIVEKIT_URL default to localhost — no changes needed
   ```

4. **Launch All Services**:
   From the project root, start everything with one command:
   ```bash
   cd ..
   ./start-all.ps1      # Windows
   ```
   Or manually start individual services:
   ```bash
   uv run langgraph dev                                   # Agent API :2024
   cd src && uv run python -m uvicorn app:app --port 7788 # Business API :7788
   ```

## 🛠️ Developer Guide

### Adding New Tools
1. Define your tool in `agent/src/tools/`.
2. Ensure it utilizes proper dependency injection from the `services/` layer if external APIs are needed.
3. Register the tool within the LangGraph nodes definition in `agent/src/logic/`.

### Extending the Voice Logic
Voice handling is centralized in `agent/src/livekit_service/`. You can customize VAD sensitivity or swap out TTS/STT providers by modifying the corresponding plugins in the voice pipeline setup.

### Using Structured Output

Get type-safe, validated responses from any LLM provider:

```python
from utils.structured_output import structured_invoke
from services.llm import get_model
from logic.schema import KnowledgeExtractionResult
from langchain_core.messages import HumanMessage

model = get_model("gpt-cloud")

# Automatic native/fallback: GPT uses native with_structured_output(),
# Ollama uses JSON prompting + Pydantic validation
result = await structured_invoke(
    model,
    [HumanMessage(content="Extract user facts from this conversation...")],
    KnowledgeExtractionResult,
    max_retries=2,
)
# result is a validated KnowledgeExtractionResult Pydantic model
for fact in result.facts:
    print(f"[{fact.category}] {fact.fact}")
```

Or use the one-liner convenience wrapper:

```python
from services.llm import get_structured_response

result = await get_structured_response(
    "deepseek", messages, KnowledgeExtractionResult
)
```

The utility automatically falls back from native structured output to JSON
parsing across providers (GPT/Cerebras → native; DeepSeek/Gemini → try
native then fallback; Ollama → always JSON).

### Manual Test Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_structured_output.py` | Test native + JSON fallback paths with real LLM |

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Built with ❤️ by [codehuang0717](https://github.com/codehuang0717)
</div>
