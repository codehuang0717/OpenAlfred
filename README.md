<div align="center">
  <img src="https://raw.githubusercontent.com/codehuang0717/OpenAlfred/main/assets/logo.png" alt="OpenAlfred Logo" width="120" />
  <h1>OpenAlfred</h1>
  <p><b>A powerful, stateful, and voice-enabled AI Agent framework</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/LangGraph-Latest-orange?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/LiveKit-Ready-brightgreen?style=flat-square" alt="LiveKit" />
    <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License" />
    <img src="https://img.shields.io/badge/Version-v1.0.1-blue?style=flat-square" alt="Version" />
  </p>
</div>

---

## 🌟 Overview

OpenAlfred is an advanced AI agent backend built on top of **LangGraph** and **LiveKit**. It is designed to be a highly modular, stateful, and voice-interactive assistant that can manage complex contexts, perform tool-based tasks, and interact with users in real-time.

Whether it's managing your reminders, holding contextual long-term conversations, or handling incoming voice calls via SIP, OpenAlfred is built for reliability and extensibility.

## ✨ Key Features

- 🧠 **Stateful Workflows**: Built with LangGraph for reliable, complex agentic logic and state management.
- 🎙️ **Real-time Voice Calls**: Seamless integration with LiveKit for low-latency voice interactions.
- 💾 **Context & Memory**: Implements intelligent context compression and long-term memory via SQLite and Mem0.
- 🛠️ **Extensible Tools**: Supports a wide range of tools including search, todo management, and system integration.
- 🚀 **High Performance**: Optimized for fast inference and streaming responses.

## 🛠️ Tech Stack

- **Core**: Python 3.13+
- **Agent Orchestration**: LangGraph, LangChain
- **Real-time Comms**: LiveKit, LiveKit Agents
- **Database**: SQLite (via aiosqlite), Mem0
- **TTS/STT**: Support for Faster-Qwen3-TTS, OpenAI Whisper, and more.
- **Dependency Management**: [uv](https://github.com/astral-sh/uv)

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) installed (recommended)
- A LiveKit server (or Cloud account)
- OpenAI or compatible LLM API keys

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/codehuang0717/OpenAlfred.git
   cd OpenAlfred/agent
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Configure Environment:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

### Running the Agent

You can start the agent using the development scripts:
```bash
# Using LangGraph CLI
langgraph dev

# Or using the start script
./start.ps1
```

## 📝 Configuration

Key settings in `.env`:
- `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `DATABASE_URL`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  Built with ❤️ by [codehuang0717](https://github.com/codehuang0717)
</div>
