<div align="center">
  <img src="https://raw.githubusercontent.com/codehuang0717/OpenAlfred/main/assets/logo.png" alt="OpenAlfred Logo" width="120" />
  <h1>OpenAlfred</h1>
  <p><b>A powerful, stateful, and modular AI Agent framework</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/LangGraph-Latest-orange?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/LiveKit-Ready-brightgreen?style=flat-square" alt="LiveKit" />
    <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License" />
    <img src="https://img.shields.io/badge/Version-v1.2.0-blue?style=flat-square" alt="Version" />
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

1. **Clone & Sync**:
   ```bash
   git clone https://github.com/codehuang0717/OpenAlfred.git
   cd OpenAlfred/agent
   uv sync
   ```

2. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Fill in your LIVEKIT_URL, REDIS_URL, OPENAI_API_KEY, etc.
   ```

3. **Launch All Services**:
   The easiest way to boot up the entire backend stack:
   ```bash
   ./start-all.ps1
   ```
   *(For manual deployment, you can start the API router and LiveKit worker separately).*

## 🛠️ Developer Guide

### Adding New Tools
1. Define your tool in `agent/src/tools/`.
2. Ensure it utilizes proper dependency injection from the `services/` layer if external APIs are needed.
3. Register the tool within the LangGraph nodes definition in `agent/src/logic/`.

### Extending the Voice Logic
Voice handling is centralized in `agent/src/livekit_service/`. You can customize VAD sensitivity or swap out TTS/STT providers by modifying the corresponding plugins in the voice pipeline setup.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Built with ❤️ by [codehuang0717](https://github.com/codehuang0717)
</div>
