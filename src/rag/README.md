# RAG — Retrieval-Augmented Generation

本地知识库模块，提供 Markdown 文档摄入（含图片解析+描述）、语义检索能力，集成到 OpenAlfred Agent 和前端。

## 架构

```
Markdown 文件
     ↓
md_parser.py     — 按标题（#/##）拆分为 Section，提取图片引用
     ↓
image_handler.py — 复制图片到 data/images/ + Gemini 生成中文描述
     ↓                    └→ image_lookup 表（id → url/alt）
chunker.py       — 按章节分块（text + heading + images 不分离）
     ↓
embedding.py     — BGE-M3 本地向量化（1024维，GPU/CUDA）
     ↓
store.py         — ChromaDB（余弦距离）+ SQLite documents 表
     ↓
retriever.py     — 语义检索，返回 chunk + 图片 URL
     ↓
tools/rag.py     — Agent 工具（search_knowledge / list_knowledge）
                    检索时 JSON 占位符解析为 ![alt](url) markdown
```

## 图片处理流水线

```
![alt](./assets/img.png)                           ← 原始 Markdown
     ↓ copy_images() → data/images/{doc_id}/
     ↓ describe_image() → Gemini 缓存到 data/descriptions/{hash}.txt
     ↓
{"_img":{"i":1,"d":"这张PPT展示了Nielsen的十条启发式原则..."}}  ← 嵌入文本
     ↓ Agent 检索时
_IMG_PLACEHOLDER regex → get_image_by_id() → ![alt](/api/images/doc_id/img.png)
```

## 文件清单

| 文件 | 作用 |
|------|------|
| `rag/embedding.py` | BGE-M3 embedding（SentenceTransformer） |
| `rag/md_parser.py` | Markdown 章节解析 + 图片引用提取 |
| `rag/image_handler.py` | 图片复制、路径重写、JSON 占位符生成 |
| `rag/image_describer.py` | Gemini 多模态图片描述 + SHA256 缓存 |
| `rag/chunker.py` | 章节分块（`chunk_sections`）+ 文本分块（`chunk_text`） |
| `rag/store.py` | ChromaDB 集合管理 + 文档存储/删除 |
| `rag/retriever.py` | 语义检索 |
| `rag/ingestion.py` | 文件/文本摄入流水线 |
| `rag/cli.py` | CLI 测试工具 |
| `rag/export_processed.py` | 导出处理后 Markdown（调试用） |
| `db/rag.py` | documents 表 + image_lookup 表 CRUD |
| `routers/rag.py` | REST API + 图片静态服务 |
| `tools/rag.py` | Agent 工具 + JSON 占位符解析 |
| `logic/prompts.py` | RAG 检索结果提示词模板 |

## 数据库

| 表 | 字段 | 说明 |
|------|------|------|
| `documents` | id, user_id, filename, title, file_type, chunk_count, created_at | 文档元数据 |
| `image_lookup` | id(INTEGER PK), document_id, url, alt, filename, created_at | 图片查表 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | embedding 模型 |
| `RAG_CHUNK_SIZE` | `500` | 分块大小（字符） |
| `RAG_CHUNK_OVERLAP` | `50` | 分块重叠 |
| `RAG_TOP_K` | `5` | 默认检索数量 |

图片描述使用 `.env` 中配置的 `GEMINI_CHAT_MODEL`（如 `gemini-3-flash-preview`）。

## 数据存储

| 数据 | 路径 |
|------|------|
| 向量数据 | `agent/chroma_db/` |
| 文档元数据 | `agent/todos.db` → `documents` + `image_lookup` |
| 图片文件 | `agent/data/images/{doc_id}/` |
| 描述缓存 | `agent/data/descriptions/{hash}.txt` |
| 模型缓存 | `~/.cache/huggingface/hub/models--BAAI--bge-m3/` |

## CLI 测试

```bash
cd agent/src

# 自测全流程
uv run python -m rag.cli demo

# 摄入文件（.md 会自动处理图片）
uv run python -m rag.cli ingest "E:\notes\笔记.md" --user <user_id>

# 摄入文本
uv run python -m rag.cli ingest-text --text "..." --title "标题"

# 语义搜索（--resolve 查看 Agent 视角完整提示词）
uv run python -m rag.cli search "查询内容" --top-k 3
uv run python -m rag.cli search "查询内容" --resolve

# 查看文档列表
uv run python -m rag.cli list

# 删除文档
uv run python -m rag.cli delete <doc_id>

# 调试模式
uv run python -m rag.cli -v search "query"

# 导出处理后 Markdown（查看嵌入内容）
uv run python -m rag.export_processed "E:\notes\笔记.md" "output.md"
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/rag/ingest-path` | 路径摄入 `{"filepath":"...", "title":""}`（推荐，自动处理图片） |
| `POST` | `/api/rag/upload` | 上传文件（multipart） |
| `POST` | `/api/rag/ingest-text` | 摄入文本 `{"title":"...", "content":"..."}` |
| `GET` | `/api/rag/documents` | 列出当前用户文档 |
| `GET` | `/api/rag/documents/{id}` | 文档详情 |
| `DELETE` | `/api/rag/documents/{id}` | 删除文档（含向量+图片） |
| `POST` | `/api/rag/search` | 语义搜索 `{"query":"...", "top_k":5}` |
| `GET` | `/api/images/{doc_id}/{filename}` | 图片静态服务（无需认证） |

## Agent 工具

| 工具 | 说明 |
|------|------|
| `search_knowledge(query, top_k)` | 语义搜索用户知识库，返回带图片的 Markdown |
| `list_knowledge()` | 列出已上传的文档 |

### 检索提示词

Agent 调用 `search_knowledge` 时，返回内容被 `RAG_SEARCH_RESULT_HEADER`（`logic/prompts.py`）包裹：

```markdown
[知识库检索结果] 以下是从用户上传的文档中检索到的相关内容。请基于这些内容回答用户问题。

规则：
1. 优先基于检索内容回答；若内容不足以回答，请明确告知而非编造
2. 引用时标注文档名和章节
3. 保留文中所有图片链接（![...](url) 语法），不要删除或修改
4. 多条检索结果时，请综合分析后给出完整回答
5. 引用块（> 开头）为图片的文字描述，可据此理解图片内容进行推理
```

## 支持的文件类型

`.md` `.txt` `.py` `.js` `.ts` `.json` `.yaml` `.yml` `.csv` `.html` `.css` `.pdf` `.docx`

> PDF 需要 `pypdf`，DOCX 需要 `docx2txt`（按需安装）。

## 前端页面

`/knowledge` — 知识库管理页面：

- **摄入**：点击选择文件 → 填目录 → 路径摄入（自动拼接文件名 + 图片处理）/ 上传摄入（不含图片处理）
- **进度**：解析 Markdown → 处理图片 & 生成描述 → 向量嵌入 & 存储 → 完成
- **文档列表**：表格展示 title/filename/chunks/date + 删除按钮
- **搜索测试**：输入查询 + top-k 选择 + RAW/Resolved 切换
- **入口**：侧边栏用户菜单 / 右面板工具箱

## 用户隔离

- 所有 RAG 数据按 `user_id` 隔离
- Web 用户通过 JWT 鉴权 → `user_id` = JWT `sub`
- CLI 用户通过 `--user` 指定，需与 Web 登录 ID 一致才能互通

## 日志

| Logger | 内容 |
|--------|------|
| `rag.embedding` | 模型加载、向量化 |
| `rag.md_parser` | Markdown 解析、Section 统计 |
| `rag.image_handler` | 图片复制、路径重写、JSON 生成 |
| `rag.image_describer` | Gemini 调用、缓存命中/写入 |
| `rag.chunker` | 章节分块、文本分块统计 |
| `rag.store` | ChromaDB/SQLite 写入、删除 |
| `rag.retriever` | 查询、结果数、分数 |
| `rag.ingestion` | 文件读取、摄入流水 |
| `rag.cli` | CLI 命令执行 |

日志文件：`agent/logs/rag.log`（`-v` 开启 DEBUG 级别）。
