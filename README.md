<div align="center">
  <img src="https://raw.githubusercontent.com/codehuang0717/OpenAlfred/main/assets/logo.png" alt="OpenAlfred Logo" width="120" />
  <h1>OpenAlfred</h1>
  <p><b>A powerful, stateful, and modular AI Agent framework</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/LangGraph-Latest-orange?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/LiveKit-Ready-brightgreen?style=flat-square" alt="LiveKit" />
    <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License" />
    <img src="https://img.shields.io/badge/Version-v1.1.0-blue?style=flat-square" alt="Version" />
  </p>
</div>

---

## 🌟 Overview

OpenAlfred is an advanced AI agent backend built on top of **LangGraph** and **LiveKit**. It features a modern, service-oriented architecture designed for reliability, low latency, and deep modularity.

Whether it's managing your reminders, holding contextual long-term conversations, or handling proactive user monitoring, OpenAlfred is built to be a robust foundation for next-generation AI assistants.

## ✨ Key Features

- 🧠 **Modular Logic**: Decoupled architecture separating core infrastructure, business services, and agent logic.
- 🎙️ **Advanced Voice Service**: Stateful voice interaction layer with built-in VAD, interrupt handling, and latency tracking.
- 💾 **Context & Memory**: Intelligent context management and long-term memory via SQLite and Mem0.
- 📡 **Event-Driven Architecture**: Redis-backed event bus for seamless inter-service communication.
- 🛠️ **Proactive Supervisor**: A unique background service that monitors context and proactively reaches out to the user.

## 📂 Project Structure

```text
agent/src/
├── core/             # Configuration, Auth, and Event Bus
├── logic/            # LangGraph Nodes, Prompts, and Graph definition
├── services/         # Business services (Email, TTS, LLM, etc.)
├── livekit_service/  # Voice interaction layer (Session, STT, Playback)
├── db/               # Database access layer
├── routers/          # FastAPI API routes
├── tools/            # Agent toolsets
└── utils/            # Shared utilities (Logging, Latency)
```

## 🛠️ Tech Stack

- **Core**: Python 3.13+
- **Agent Orchestration**: LangGraph, LangChain
- **Real-time Comms**: LiveKit, LiveKit Agents
- **Event Bus**: Redis Pub/Sub
- **Database**: SQLite (aiosqlite), Mem0
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)

## 🚀 Getting Started

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

2. **Configure**:
   ```bash
   cp .env.example .env # Fill in your credentials
   ```

3. **Launch All Services**:
   ```bash
   ./start-all.ps1
   ```

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Built with ❤️ by [codehuang0717](https://github.com/codehuang0717)
</div>
