# OpenAlfred — AI Study Assistant & Proactive Personal AI Agent

[简体中文](README.md) · English · [Agent-readable overview](llms.txt)

**OpenAlfred is an open-source, self-hosted AI study assistant for students and self-learners, combining a proactive personal AI agent, long-term memory, multimodal RAG, voice calls, and AI-generated mini-apps.** It helps students and individuals connect their notes, tasks, reminders, and email in one personal AI assistant — a practical JARVIS-style assistant for study and everyday routines.

Built with **Python, FastAPI, LangGraph, and LangChain**, with a **Next.js / React** frontend. Use configured cloud LLMs or local models through **Ollama**.

## See it in 84 seconds

https://github.com/user-attachments/assets/31267eff-2887-4f7c-a29a-ec9956534ad5

The video shows proactive reminders, cross-session memory, answers with original note images, and generated tools returning to the toolbox. The browser and product UI are actual captures; the incoming-call sequence is a staged demonstration.

## AI for students and self-learners

OpenAlfred is useful as a **student AI assistant**, **study companion**, and **student productivity assistant** for university coursework, independent learning, and postgraduate note organization.

- **Lecture notes assistant / PDF Q&A:** ask questions against your own Markdown and PDF course notes, with source text and original illustrations.
- **Revision assistant / exam preparation:** revisit concepts using your notes and remembered explanation preferences.
- **Study planner / assignment deadline reminders:** keep coursework, course projects, and dissertation milestones in a task-and-reminder workflow.
- **Focus assistant / study accountability:** opt into supervision that relates recent screen activity to the task you intended to work on.
- **AI study tools:** generate a Pomodoro timer, calculator, or progress tracker and keep it in your toolbox.
- **Academic email workflow:** find and summarize school or supervisor emails, then review reply drafts before sending.

## What makes it a proactive AI assistant?

Instead of requiring a new chat message for every interaction, OpenAlfred can schedule reminders and, when the user enables supervision, use task context and recent screen activity to decide when to reach out. Notifications and telephone reminders require their respective services to be configured.

## Capabilities

| Looking for | What OpenAlfred provides |
|---|---|
| **Proactive AI agent / agentic personal assistant** | Task-aware reminders, optional screen supervision, push notifications, and outgoing calls. |
| **Long-term memory / personalized AI assistant** | User-scoped facts, preferences, habits, and relationship notes reused across conversations. |
| **Multimodal RAG / personal knowledge base** | Retrieval-augmented generation over Markdown and PDF notes, including source references and original note images. |
| **Voice agent / conversational voice assistant** | Speech-to-text → LLM → text-to-speech, with LiveKit and SIP for calls. |
| **Self-hosted AI / local LLM assistant** | Run the application and storage yourself; choose configured cloud providers or local Ollama models. |
| **Generative UI / AI mini-app generator** | Generate standalone HTML, CSS, and JavaScript tools from natural-language requirements; preview before publishing to the personal toolbox. |
| **Email assistant / productivity assistant** | IMAP / SMTP email workflows, reply drafts, todos, and scheduled reminders. |
| **AI study assistant / study planner** | Combine personal course notes, remembered learning preferences, and study tasks. |
| **AI image generation** | Request a new image through a configured image model and display the result in chat. This is separate from retrieving an original image from notes. |

Examples:

- “Use my PCA notes to explain the derivation, with the original illustration.”
- “Remind me before Friday's assignment deadline.”
- “Summarize my recent email and draft a reply for me to confirm.”
- “Create a focus timer and keep it in my toolbox.”

## Architecture

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph, LangChain |
| Backend API | FastAPI, Python, SQLite |
| Web interface | Next.js, React, Tailwind CSS, shadcn/ui |
| Personal knowledge retrieval | BGE-M3 embeddings, ChromaDB, document and image processing |
| Events and scheduling | Redis Pub/Sub and delayed queues |
| Voice | LiveKit Agents, SIP, configured STT and TTS services |
| Local models | Ollama integration |

Configured model integrations include OpenAI-compatible endpoints, DeepSeek, Gemini, Cerebras, and Ollama. Individual features depend on the provider's supported capabilities and the user's configuration.

## Deployment and boundaries

- Install backend dependencies with `uv sync`; use `start-all.ps1` to launch the configured Windows service stack. See the [quick-start overview](README.md#快速开始) and the separate frontend repository for web setup. The frontend runs at `http://localhost:3000`.
- This repository contains the Python backend; the web interface is maintained separately at [OpenAlfred-WebUI](https://github.com/codehuang0717/OpenAlfred-WebUI).
- Windows desktop capture and voice components have platform-specific requirements. Self-hosting does not make every feature available on every operating system.
- Selecting a cloud model sends the relevant model requests to that provider. Self-hosting alone does not imply fully offline inference.
- Standalone generated mini-apps run in a restricted browser preview and cannot directly access private mail, memory, the host filesystem, or the network. A separate todo-timeline tool provides the supported personal-data panel.
- Email sending requires user confirmation. Voice calls, push notifications, screen supervision, and image generation require configuration.

## Source entry points

For developers and agents inspecting the implementation:

- [Agent tool registry](src/tools/__init__.py)
- [Long-term memory tools](src/tools/memory.py)
- [Knowledge retrieval and original-image resolution](src/tools/rag.py)
- [Task and reminder tools](src/tools/todos.py)
- [Telephone integration](src/tools/call_user.py)
- [Generated mini-app tools](src/tools/user_apps.py)
- [Mini-app generation and validation](src/services/code_apps.py)
- [Image generation tool](src/tools/image_generation.py)
- [LLM provider integrations](src/services/llm.py)

## License

The project README declares the [MIT license](README.md#license).
