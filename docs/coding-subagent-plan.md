# Coding Subagent 实现方案

状态：2026-09-30 已实现第一版（任务底座、Coding Agent 和前端任务卡）；浏览器功能验收及真实 API 基准待完成。

用户确认允许 Coding Agent 继承私人数据工具。实际继承**只读白名单**，由后端绑定账号与时区；不继承修改待办、发送邮件、记忆写入、电话、任意浏览器或本机 shell 工具。生成代码可以包含所需数据快照，但不获得动态账号访问能力。

## 决策摘要

保留现有 LangGraph 主 Agent，增加独立上下文、显式工具白名单的 Coding Agent。使用已经引入的 LangChain `create_agent()` 承担模型/工具循环，LangGraph 承担执行状态与 checkpoint；业务代码只负责用户授权、工作区、任务生命周期、验证和版本发布。第一版不替换主 Agent，不同时引入另一套通用 Agent 调度框架。

采用 manager 模式：主 Agent 理解需求并提交后台编码任务，仍负责聊天；Coding Agent 不接管整个会话。任务立即返回 `job_id`，进度和最终结果通过持久化任务卡展示。不能把阻塞的子图调用改个名字就当作后台任务。

首期范围仍是无外部依赖的 HTML/CSS/JavaScript 小程序，不是任意 Python/Node 服务或完整仓库开发。自动生成只产生可预览草稿，发布仍需用户确认。

## 改造前的核查结果

- `src/services/code_apps.py` 的 `write_code_source()` 一次要求完整 JSON，校验失败后最多整份重写一次；没有读取、局部编辑和基于实际运行结果修复的工具循环。
- `src/tools/user_apps.py` 的创建工具等待整个生成结束。响应式修订则使用 FastAPI `BackgroundTasks`，两条路径的任务运行机制不同。
- 模型输出已通过 `callbacks=[]` 与主聊天流隔离，应保留该边界，不能为了显示进度再次把源码转发到聊天气泡。
- `node --check -` 只检查语法；正则规则不能证明交互正确，也不是恶意代码隔离。
- 现有用户应用、修订、任务表与发布 API 可以复用。原来的真实待办时间线使用受信任组件，应保留；新生成代码不获得运行时账号权限。

## 主流框架对比

| 框架 | 已核对的模式 | 对本项目的判断 |
| --- | --- | --- |
| [LangGraph / Deep Agents](https://docs.langchain.com/oss/python/deepagents/subagents) | 独立子 Agent 上下文、专用工具和模型；[异步任务](https://docs.langchain.com/oss/python/deepagents/async-subagents)支持启动、查询、更新与取消 | 与现有技术栈最贴近。采用其隔离与任务契约；首期用 LangChain/LangGraph 现有基础能力实现受限 Coding graph |
| [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents/orchestration) | agents-as-tools 保留 manager 的回复责任；handoff 则转移会话控制 | 借鉴 manager 模式，不为一个编码功能迁移整个主 Agent |
| [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/subagents) | 每个子 Agent 的专用提示、独立会话、工具限制与简短结果回传 | 借鉴上下文和权限边界；不把当前多供应商应用改成 Claude SDK 专属运行路径 |
| [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) | 文件编辑、终端等编码工具与执行环境；[TaskToolSet](https://docs.openhands.dev/sdk/guides/task-tool-set)委派并保留可恢复任务上下文 | 更适合未来完整工程开发。当前 TaskToolSet 是阻塞委派，不能直接解决聊天等待；引入前还需处理 Windows 执行环境、工具收权及双运行时成本 |

这是针对当前仓库的工程取舍，不是框架能力排名。[Deep Agents 架构](https://github.com/langchain-ai/deepagents/blob/main/libs/ARCHITECTURE.md)本身也基于 LangChain 的 `create_agent()` 与 LangGraph；不采用其整套默认 harness，并不意味着重新手写模型/工具循环。

## 运行链路

```text
主 Agent 收集需求 → 创建持久化 job → 立即回复已开始
                                    ↓
                            后台 Coding graph
                                    ↓
                       读写文件 → 验证 → 按反馈修复
                                    ↓
                       草稿 + 验证报告 + 最终任务卡
                                    ↓
                            用户预览并手动发布
```

### 上下文与模型

- 子 Agent 收到本次需求、验收条件、提交时间、用户时区、平台限制及指定版本源码；不复制整个聊天。按用户需求从本人待办、邮件、记忆和知识库读取必要数据，不提前全量灌入。
- 独立的消息历史、预算、checkpoint 和 trace；由后端生成任务身份，范围包含 `user_id/app_id/job_id`。每次读取仍须做所有权检查，ID 不等于授权。
- 主聊天只保存任务 ID、状态、简短报告和版本引用。源码留在工作区/修订中，按文件或行范围读取，避免每轮复制整份代码。
- 统一使用严格的模型供应商适配入口，消除现有编码模型工厂的重复配置。提交时记录实际供应商和模型配置快照；缺少配置、模型不支持工具或 API 失败均明确报错，不切模型、不回退到活跃用户或 `default`。
- 编码预检查和每个任务显式拥有独立 HTTP 客户端；不能关闭 LangChain 默认复用的共享 transport，否则会破坏主聊天/摘要/标题模型的后续调用。正常主模型仍复用其长生命周期连接池。
- 第一版只有一个 Coding Agent，不再嵌套生成 reviewer、tester 等 Agent；验证由确定性的工具承担。

### 工具与工作区

建议工具：`list_files`、`read_file`、`write_file`、`apply_patch`、`validate_app`、`finish_task`。工作区固定为任务内的 `index.html`、`styles.css`、`app.js`，由服务映射回现有 `html/css/javascript` 修订格式。

文件是用户隔离的逻辑文件，不授予主仓库或本机目录访问。禁止任意 shell、安装依赖、读取 `.env`、启动服务和直接发布。单文件最多 30,000 字符，补丁校验基准 hash；源文件修改直接写入 LangGraph checkpoint state，不产生重复宿主机文件操作。模型参数不能覆盖后端绑定的用户身份。

继承 `get_todos`、`get_email_accounts`、`get_recent_emails`、`read_email`、`get_user_profile`、`get_user_memory_category`、`list_knowledge`、`search_knowledge`。邮件读取必须给出具体账号，拒绝回退到第一个账号。数据结果存入子任务只读 `data/` 逻辑文件，长结果只回传短预览，`read_file` 可按行和字符分页读取完整内容。单次原始结果最多 200,000 字符，任务累计上限 600,000 字符。

验证后返回结构化问题：文件、位置、错误类型、可修复建议。只有指定源码 hash 的验证报告通过，`finish_task` 才能提交草稿；“模型说完成”、空回复、截断或只调用工具不构成完成。平台错误保留具体错误类型，不能伪装成验证成功。

### 任务生命周期与交付

- 创建和修订统一进入现有任务表；状态为 `queued/generating/ready/failed/cancelled/interrupted`，阶段另记 `planning/coding/validation/repairing`。
- 第一版由 FastAPI 生命周期管理一个有界异步 worker 池，调用独立 Coding graph；数据库事务领取任务、租约与 heartbeat 防止重复领取，不为每个任务启动 Python 进程。后续可替换为独立 worker，任务协议不变。
- 本地使用明确配置的持久化 LangGraph checkpointer；业务任务表存状态、运行 ID、事件序号和取消代次。不能把 `langgraph dev`、内存 checkpointer 或 `BackgroundTasks` 当作重启恢复保证。
- 服务退出或租约失效后标记 `interrupted`，保留 checkpoint；用户显式恢复/重试，不能偷偷重放写入或继续付费调用。没有有效 checkpoint 时明确说明需重新生成。
- 删除应用或取消任务先使当前执行代次失效，再停止执行；完成写入同时检查所有权、任务状态、代次和应用存在性，拒绝过期结果。删除同时清除各次重试的私人数据 checkpoint；若删除期间服务崩溃，下次启动清理孤儿 checkpoint。
- 每个任务保存来源会话。事件持久化，API 支持序号补读与 SSE；当前前端每 1.5 秒读取任务事件和状态快照，终态停止。切换会话不取消生成，回到原会话或刷新可重新读取最终报告。
- 主 Agent 提交成功后只承诺“已开始”，不循环轮询。完成时任务卡展示短报告、失败原因或草稿入口，不依赖主聊天还在流式输出，也不伪造主模型最终回答。
- 默认每用户 1 个、全局 2 个；模型调用 24 次、验证失败后最多 3 次修复、单次输出 16,000 token、总等待 900 秒。输入/累计 token 限制默认关闭（配置为 `0`），正数可显式启用。统计使用与主会话相同的 `cl100k_base` 协议消息与完整工具 schema 估算，包含 MiMo 推理字段，不重复计入 SDK 内部元数据；非该 tokenizer 的供应商需按真实 usage 校准。兼容字段 `token_upper_bound` 保存估算输入加输出预留累计值，不是供应商精确计费量。调用前持久化预留预算，恢复累计；明确重新生成才开启新预算。调用数或时长超限显式失败，不无限摘要或重试。

## 验证和执行隔离

基础版本只在后端解析源码，不运行生成的 Python/Node 代码；iframe 无同源权限，CSP 限制连接与资源加载，不传主站令牌。账号数据仅通过用户批准的生成时快照进入源码。前后端补充常见网络/导航、表单、内联事件和 HTML 字符串注入拒绝规则；这些静态规则不是对混淆恶意代码的安全证明。报告明确 `browser_tests: not_run`。

第二阶段增加独立、无登录态的浏览器验收：窄栏和宽屏尺寸、控制台错误、关键按钮、布局溢出、canvas 清晰度与动画。验证页面阻断外部网络、下载及主站能力，设置超时并终止专用浏览器进程；只发送限定大小的日志/截图给编码 Agent。

上下文隔离、文件白名单与浏览器预览均不等于操作系统级执行沙箱。本机浏览器验收只能作为受控开发验证，不能承诺恶意代码的可靠 CPU/内存隔离。若后续开放服务端代码、包安装或多租户不可信代码执行，必须先选择真正的执行隔离方案；不以开放本机 shell 代替沙箱。[Deep Agents 的后端文档](https://docs.langchain.com/oss/python/deepagents/backends)也明确区分文件访问限制与宿主机 shell 权限。

## 改造顺序与兼容性

1. **任务与隔离底座**：统一创建/修订队列、checkpoint、取消、删除防竞态、用户权限和结构化进度；先用假模型测试生命周期。
2. **Coding Agent**：接入 `create_agent()`、受限文件工具和完成契约，替换一次性 JSON 生成。移除旧的整份重写循环及重复模型工厂，不保留静默降级路径。
3. **前端交付**：任务卡、阶段进度、取消/恢复与终态报告；复用侧栏草稿、源码查看、放大预览、发布和删除。
4. **功能验收与评测**：增加浏览器验证闭环，完成同模型真实 API 基准后再考虑更复杂项目或 OpenHands runtime。

应用 ID、版本和受信任时间线保持兼容，新 Agent 失败不覆盖已发布源码。旧版本若使用现在禁止的内联事件、HTML 注入或导航 API，预览会显示具体安全规则错误，仍可查看旧源码并显式点击“生成兼容草稿”；发布前按当前规则再次验证。不偷偷修改或发布旧源码。当前不支持持久化程序状态。新增配置写入 `.env.example`，不提交凭据、checkpoint 数据或个人信息。

## 测试与指标

- 固定功能集：番茄钟、计算器、2D 飞机动画、示例数据时间线，以及在现有版本上增加功能；覆盖保持旧功能和局部修改。
- 隔离：跨用户读写/查询/取消/发布拒绝；路径穿越、绝对路径和 Windows UNC 拒绝；生成代码不获得主站凭据或其他用户数据。
- 模型：工具错误可回传修复，空输出/截断/非法工具/API 错误不成功，预算超限可见，无隐藏模型切换。
- 生命周期：切换会话、刷新/SSE 补读、重启、重复领取、取消与删除竞态、过期完成、checkpoint 恢复和已有发布版本不变。
- 浏览器：320px/宽屏适配、基础交互、console 错误、外网阻断、死循环时终止验证；语法通过不能冒充功能通过。
- 比较当前一次性生成与新 Agent：成功率、修复轮数、首次状态延迟、生成完成时间、输入/输出/缓存 token、主会话上下文增量、进程数和峰值内存。子 Agent 不保证总 token 更少；目标是降低主会话负担并提升可验证成功率。供应商未提供的指标记为不可用，不记为 0。
- 自动化测试 mock 外部服务；真实 API 放在手动 probe，记录模型配置与重复次数，不使用真实私人数据或调用订阅绕过接口。

## 实施边界

阶段 1–3 已落地，自动化测试使用真实 LangChain/LangGraph 工具循环和 SQLite checkpoint，但模型及外部 API 为测试替身；不代表真实模型的生成质量已验证。新增依赖为 `langgraph-checkpoint-sqlite`，不另建通用 Agent 运行框架。阶段 4 尚未实现，不能声称语法验证等于交互验收，或 iframe 提供可靠 CPU/内存/网络安全隔离。
