<h1 align="center">OpenAlfred</h1>
<p align="center">
  <strong>面向学生与自学者的开源 AI 学习助手：主动提醒、记住学习偏好、结合笔记原图答疑</strong><br>
  主动提醒 · 长期记忆 · 图文 RAG 知识库 · 语音 Agent · 按需生成小程序
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <a href="README.en.md">English</a> · <a href="llms.txt">Agent-readable overview</a>
</p>

**OpenAlfred is an open-source, self-hosted AI study assistant and proactive personal AI agent for students and self-learners**, with long-term memory, multimodal RAG, voice calls, email and task management, and AI-generated mini-apps. Built with LangGraph and FastAPI, it supports cloud LLMs and local models through Ollama.

<p align="center">
  <img src="https://img.shields.io/badge/backend-FastAPI_+_LangGraph-009688?logo=fastapi" alt="Backend">
  <img src="https://img.shields.io/badge/frontend-Next.js_16-000000?logo=next.js" alt="Frontend">
  <img src="https://img.shields.io/badge/Python-3.13+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

## 56 秒认识 OpenAlfred

https://github.com/user-attachments/assets/53de2afe-2462-4919-a51d-6475712ece85

主动来电提醒、跨会话记忆、笔记原图引用，以及按需生成并留在工具箱里的小程序。

> 网页与产品界面为实际录制；来电部分为场景演绎。

---

## 它不是聊天机器人

市面上大多数 AI 助手是这样的：你问，它答。你忘了问，它就沉默。

**OpenAlfred 不一样：**

<table>
<tr>
<td width="33%" align="center">
  <h3>🔔 主动型</h3>
  <p align="left">不等你开口。后台监测你的学习状态，到时间了打电话/推送提醒你。检测到分心——直接一通电话打过来。</p>
</td>
<td width="33%" align="center">
  <h3>🧠 带记忆</h3>
  <p align="left">不是金鱼。它会从每段对话中提取关于你的事实、偏好和习惯，存成长期记忆。你提过的东西，下一次它会记得。</p>
</td>
<td width="33%" align="center">
  <h3>🎓 为学生而生</h3>
  <p align="left">上传课堂笔记、PPT 截图、教科书 PDF。Alfred 识别文档里的图片并生成描述，回答时精准引用原文和插图。</p>
</td>
</tr>
</table>

---

## 为学生与自学者设计（AI for Students）

如果你在找 **学生 AI 助手、大学生学习助手、AI study assistant、study companion**，OpenAlfred 把个人笔记问答、学习任务和主动提醒放进同一个助理。适用于课程学习、自学与复习，也可用于研究生的笔记整理和任务安排。

| 学习场景 | 对应功能与英文检索词 |
|---|---|
| 课件和课堂笔记太多，想按自己的资料提问 | 上传 Markdown / PDF，检索原文与插图：**lecture notes assistant**, **PDF Q&A**, **course notes**, **personal knowledge base** |
| 复习时想按自己的理解顺序讲解 | 结合已有笔记、原图和记住的偏好回答：**revision assistant**, **exam preparation**, **personalized learning** |
| 作业、课程设计和论文节点容易忘 | 管理任务与定时提醒：**study planner**, **assignment deadline reminder**, **student productivity**, **academic task management** |
| 打开电脑准备学习，结果一直刷视频 | 用户开启监督后，结合任务和屏幕活动提醒回到学习：**focus assistant**, **study accountability**, **proactive study reminders** |
| 想给自己做一个小工具 | 用自然语言生成番茄钟、计算器或进度追踪器：**AI study tools**, **Pomodoro timer**, **AI-generated mini-apps** |
| 导师或学校邮件容易漏看 | 查询邮件、整理内容并起草回复：**student email assistant**, **academic email workflow** |

例如：“结合我上传的 PCA 笔记和原图，按推导顺序解释一遍。”“周五课程设计截止，提前提醒我。”“帮我做一个复习用的番茄钟。”

## 你在寻找这样的 AI Agent 吗？

OpenAlfred 面向希望自己部署个人 AI 助理、构建学习助手，或研究主动式 Agent 工作流的开发者。以下是常见的中英文检索方式，以及对应的实际功能：

| 方向与常见关键词 | OpenAlfred 中的对应能力 |
|---|---|
| 主动式智能体 / **proactive AI agent**, **agentic AI**, personal AI assistant, JARVIS-style assistant | 在用户开启监督并配置提醒方式后，结合任务与屏幕活动触发提醒、推送或电话 |
| 记忆型助手 / **long-term memory**, memory agent, personalized assistant | 保存用户偏好、事实与习惯，在后续会话中使用 |
| 个人知识库 / **multimodal RAG**, retrieval-augmented generation, personal knowledge base | 检索 Markdown / PDF 笔记，返回文字、来源和笔记原图 |
| 语音智能体 / **voice agent**, voice assistant, STT / TTS, LiveKit, SIP | 通过语音识别、LLM 与语音合成完成双向通话；电话服务需另行配置 |
| 自托管助手 / **self-hosted AI assistant**, Ollama, local LLM | 自己运行应用与数据存储，选择本地或云端模型；选用云模型时相应请求仍会发往提供商 |
| 生成式界面 / **generative UI**, AI mini-app generator, natural-language-to-app | 用自然语言生成计时器、计算器、小游戏等独立小程序，预览并确认发布到个人工具箱 |
| 日常效率 / AI productivity assistant, email assistant, task management | 待办、定时提醒、IMAP / SMTP 邮件读取与回复草稿 |
| 学习助手 / **AI study assistant**, student AI assistant, study planner | 结合个人笔记、学习偏好与任务安排辅助复习和管理截止日期 |

这些方向描述项目用途；独立生成的小程序没有直接读取邮件、记忆或文件系统的权限。真实待办时间线使用单独的数据面板工具。

## 核心能力

### 🔔 主动来找你，不用你去找它

| 场景 | Alfred 怎么做 |
|---|---|
| 「明天早上提醒我吃饭」 | 到时间 Bark 推送到手机|
| 你在刷视频而不是写论文 | 屏幕监控检测到分心 → 累计到阈值 → 来电提醒 |
| 待办任务逾期未完成 | 主动催促，逐步升级提醒强度 |

<p align="center">
  <img src="assets/screenshots/reminder.png" width="70%" alt="提醒">
</p>
<p align="center">
  <img src="assets/screenshots/barknotification.png" width="35%" alt="推送通知">
</p>

> 后台运行一个 **Supervisor（主管进程）**，定期截屏 + OCR → LLM 判断当前活动是否与任务相关。检测到你在摸鱼，它真的会打电话给你。

### 🧠 长期记忆，越用越懂你

每轮对话结束后，Alfred 自动从聊天中提取关于你的信息：

- **profile** — 姓名、学校、专业、重要日期
- **preferences** — 喜欢/讨厌的事物、学习偏好
- **patterns** — 作息习惯、工作方式
- **relationship** — 与导师、同学、家人的互动

这些记忆以 Markdown 文件存储在 `memory/{你的用户ID}/` 目录下，**完全透明、可手动编辑**。下次对话时自动注入上下文，Alfred 会基于对你的了解做出更精准的回应。

```
memory/
├── 550e8400-.../
│   ├── profile.md         # "用户是XX大学计算机专业研二学生"
│   ├── preferences.md     # "喜欢用费曼学习法，偏好早上6-8点深度学习"
│   ├── patterns.md        # "每周三下午有组会，周五晚上打篮球"
│   └── relationship.md    # "导师姓王，对论文格式要求严格"
```

### 🎓 学生场景深度优化

#### 笔记上传，图片不用愁

上传你的 Markdown 笔记或 PDF 课件——Alfred 会自动：

1. **按标题分段** — 保留文档结构
2. **复制并描述图片** — 笔记里的截图、图表、公式，由 Gemini 多模态模型生成文字描述
3. **向量化存储** — BGE-M3 嵌入 → ChromaDB

提问时，Alfred 检索相关内容并**引用原文 + 还原图片**：

<p align="center">
  <img src="assets/screenshots/knowledgebase.png" width="70%" alt="知识库检索">
</p>

> 「根据我的操作系统笔记，虚拟内存和物理内存的区别是什么？」→ Alfred 找到你笔记中的相关章节，文字 + 原图一起返回。

#### 学习任务管理

```
你：帮我记一下，下周五之前要交编译原理的课程设计
你：每周一到周五早上8点提醒我背150个GRE单词
你：列出我这周所有还没完成的任务
你：最近我编译原理的任务进度怎么样
```

<p align="center">
  <img src="assets/screenshots/task.png" width="60%" alt="任务管理">
</p>

---

## 更多能力

### 📞 语音通话

SIP 协议接入，支持双向通话：

- Alfred 主动打给你（提醒/分心警告）
- 你打给 Alfred（在路上、在厨房，不方便打字时）

语音管线：SenseVoice STT → LLM → Faster-Qwen3-TTS

<p align="center">
  <img src="assets/screenshots/callin.jpg" width="40%" alt="来电">
  <img src="assets/screenshots/calling.png" width="40%" alt="通话中">
</p>

### 📧 邮件

配置邮箱后：

- 「总结一下导师最近发来的邮件」
- 「帮我看一下上周那封关于开题报告的邮件」
- 起草回复 → 你确认 → 发送（LLM 不能直接发，必须你点确认）

<p align="center">
  <img src="assets/screenshots/summaryemail.png" width="45%" alt="邮件总结">
  <img src="assets/screenshots/draftemail.png" width="45%" alt="邮件草稿">
</p>

### 🧩 自然语言生成个人工具（Generative UI）

描述你需要的工具，例如番茄钟、计算器、进度追踪器或小游戏，Alfred 会生成 HTML / CSS / JavaScript 小程序。先查看草稿预览，确认后再发布到右侧工具箱；已有工具可继续打开使用。

这适合 **AI-generated apps**、**mini-app generation** 和 **natural-language UI generation** 场景。生成代码在受限的浏览器预览中运行，不会在宿主机上直接执行；它与使用真实待办的数据面板是两类工具。

### 🎨 AI 图片生成与笔记原图引用

除了从知识库引用已有图片，Alfred 也提供 **AI image generation** 工具，按文字要求调用已配置的图像模型生成新图片，并在聊天中展示。生成新图与检索笔记原图是独立能力。

### 🤖 多模型切换

| 模型 | 适合场景 |
|---|---|
| GPT 系列 | 综合能力最强，复杂推理 |
| DeepSeek V4 | 性价比高，中文理解好 |
| Gemini 2.5 Flash | 多模态、图片理解 |
| Cerebras Llama | 极速推理 |
| 本地 Gemma（Ollama） | 隐私优先、离线可用 |

### 🔔 Bark 推送

在 iOS 上装 Bark App，填入设备 URL，Alfred 就能推送到手机。

<p align="center">
  <img src="assets/screenshots/barktest.png" width="70%" alt="Bark 测试">
</p>
<p align="center">
  <img src="assets/screenshots/barknotification.png" width="35%" alt="通知效果">
</p>

---

## 快速开始

### 你需要

- Python 3.13+ + [`uv`](https://docs.astral.sh/uv/)
- Node.js 20+
- **Redis**（事件总线 + 延迟队列）
- **LiveKit Server**（语音通话，即使不用语音也要装）
- 至少一个 LLM API Key

### 三步跑起来

```bash
git clone https://github.com/codehuang0717/OpenAlfred.git
cd OpenAlfred

# 1. 安装依赖
./setup.ps1      # Windows
# ./setup.sh     # Linux / macOS

# 2. 编辑 agent/.env，填入 API Key
# OPENAI_API_KEY=sk-xxx

# 3. 启动
./start-all.ps1
```

打开 **http://localhost:3000**，注册账号，开始用。

### 安装 Redis

```bash
# Windows: winget install Redis.Redis
# macOS: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo systemctl enable redis-server --now

redis-cli ping  # → PONG 说明装好了
```

### 安装 LiveKit

从 [LiveKit Releases](https://github.com/livekit/livekit/releases) 下载对应平台的二进制文件，放到 `bin/` 目录下。启动脚本会自动检测。

### 可选：语音 Docker 服务

```bash
docker compose up -d sensevoice                    # STT
docker compose --profile gpu build tts             # TTS（需 GPU）
docker compose --profile gpu up -d tts
```

---

## 架构

```
┌─────────────────── 你的电脑 ────────────────────────────────┐
│  Redis :6379    LiveKit :7880    Ear Service (唤醒词监听)   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Agent (Python)          WebUI (Next.js :3000)       │  │
│  │  LangGraph :2024         ├ 对话界面                   │  │
│  │  FastAPI :7788           ├ 设置中心                   │  │
│  │  Worker (后台调度)       ├ 知识库管理                 │  │
│  │  Supervisor (屏幕监控)   └ 邮件面板                   │  │
│  │  LK Voice Workers (:5883, :5884)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─── Docker (可选) ────────────────────────────────────────────┐
│  SenseVoice STT :8000    Faster-Qwen3-TTS :7017 (GPU)      │
└─────────────────────────────────────────────────────────────┘
```

| 层 | 技术栈 |
|---|---|
| Agent | Python + FastAPI + LangGraph + aiosqlite |
| 前端 | Next.js 16 + React 19 + Tailwind CSS 4 + shadcn/ui |
| 语音 | LiveKit Agents (STT → LLM → TTS) + SIP |
| 知识库 | BGE-M3 向量化 + ChromaDB + Gemini 多模态图片描述 |
| 事件 | Redis Pub/Sub + Sorted Set 延迟队列 |

---

## 项目结构

```
├── agent/               # Python 后端
│   └── src/
│       ├── routers/     # 11 个 API 路由
│       ├── tools/       # 10+ 组 Agent 工具
│       ├── logic/       # LangGraph 状态图 + 提示词
│       ├── services/    # LLM / 邮件 / 通知 / 调度器
│       ├── db/          # aiosqlite (WAL mode)
│       ├── rag/         # 知识库管线（解析→图片描述→分块→向量化→检索）
│       └── livekit_service/  # 语音 Agent
├── web/                 # Next.js 16 前端
├── memory/              # L1 长期记忆（每个用户一个目录）
├── assets/              # TTS 参考音频 + 截图
├── docker-compose.yml
└── setup.ps1 / start-all.ps1
```

---

## License

MIT

---

<p align="center">
  <sub>Built by <a href="https://github.com/codehuang0717">codehuang0717</a></sub>
</p>
