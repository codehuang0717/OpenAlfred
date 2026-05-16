# RAG — Retrieval-Augmented Generation

本地知识库模块，提供文档摄入、语义检索能力，集成到 OpenAlfred Agent 中。

## 架构

```
用户上传文件/文本
       ↓
  ingestion.py    — 文件解析（txt/md/py/pdf/docx）
       ↓
  chunker.py      — 文本分块（RecursiveCharacterTextSplitter）
       ↓
  embedding.py    — BGE-M3 本地向量化（1024维，GPU推理）
       ↓
  store.py        — ChromaDB 向量存储 + SQLite 元数据
       ↓
  retriever.py    — 语义检索（余弦相似度）
```

## 依赖

| 组件 | 用途 | 模型/配置 |
|------|------|-----------|
| `sentence-transformers` | 本地 embedding | `BAAI/bge-m3` (2.2GB, 首次下载缓存) |
| `chromadb` | 向量存储 | 持久化到 `agent/chroma_db/` |
| `langchain-text-splitters` | 文本分块 | chunk_size=500, overlap=50 |
| SQLite (`aiosqlite`) | 文档元数据 | `documents` 表 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | embedding 模型 |
| `RAG_CHUNK_SIZE` | `500` | 分块大小（字符） |
| `RAG_CHUNK_OVERLAP` | `50` | 分块重叠 |
| `RAG_TOP_K` | `5` | 默认检索数量 |

## CLI 测试

```bash
cd agent/src

# 自测全流程
uv run python -m rag.cli demo

# 摄入文件
uv run python -m rag.cli ingest /path/to/doc.txt --title "我的笔记"

# 摄入文本
uv run python -m rag.cli ingest-text --text "OpenAlfred 是一个开源AI助手..." --title "简介"

# 语义搜索
uv run python -m rag.cli search "什么是 OpenAlfred" --top-k 3

# 列出文档
uv run python -m rag.cli list

# 删除文档
uv run python -m rag.cli delete <doc_id>

# 开启调试日志
uv run python -m rag.cli -v search "query"
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/rag/upload` | 上传文件（multipart） |
| `POST` | `/api/rag/ingest-text` | 摄入文本 `{"title":"...", "content":"..."}` |
| `GET` | `/api/rag/documents` | 列出文档 |
| `GET` | `/api/rag/documents/{id}` | 查看文档详情 |
| `DELETE` | `/api/rag/documents/{id}` | 删除文档 |
| `POST` | `/api/rag/search` | 语义搜索 `{"query":"...", "top_k":5}` |

## Agent 工具

Agent 自动拥有以下工具：

- **`search_knowledge(query, top_k)`** — 搜索用户知识库，返回相关片段及来源
- **`list_knowledge()`** — 列出已上传的文档

## 支持的文件类型

`.txt` `.md` `.py` `.js` `.ts` `.json` `.yaml` `.yml` `.csv` `.html` `.css` `.pdf` `.docx`

> PDF 需要 `pypdf`，DOCX 需要 `docx2txt`（按需安装）。

## 日志

使用项目统一的 `utils.logger`，logger 名称前缀 `rag.`：

| Logger | 内容 |
|--------|------|
| `rag.embedding` | 模型加载、向量化过程 |
| `rag.chunker` | 分块结果统计 |
| `rag.store` | ChromaDB/SQLite 写入、删除 |
| `rag.retriever` | 查询、结果数、相关性分数 |
| `rag.ingestion` | 文件读取、文本提取、摄入流水 |
| `rag.cli` | CLI 命令执行 |

日志文件：`agent/logs/rag.log`（通过 CLI `-v` 可开启 DEBUG 级别）。

## 数据存储

- **向量数据**: `agent/chroma_db/`（ChromaDB 持久化）
- **元数据**: `agent/todos.db` → `documents` 表
- **模型缓存**: `~/.cache/huggingface/hub/models--BAAI--bge-m3/`
