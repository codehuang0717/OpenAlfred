# 聊天图片生成

聊天中的 GPT 选项使用 `CLOUD_CHAT_MODEL=gpt-6-sol`，通过 Responses API 支持推理和工具调用。实际绘图由 `generate_image` 工具调用 Images API 完成，模型由 `IMAGE_GENERATION_MODEL` 指定，默认 `gpt-image-2.5-flare`，沿用 `OPENAI_API_KEY`。

重启后端和 LangGraph 服务后，在聊天中选择 GPT，输入例如：

> 生成一张水彩风格的海边日落图片，横向构图。

工具完成后图片直接显示在回复中，点击可放大，刷新聊天后仍能查看。当前工具支持生成新图；修改已有图片需要单独的编辑接口。

## 存储与错误处理

- PNG 存放在 `data/generated_images/<账号哈希>/`，不进入公共静态目录。
- `/api/generated-images/{image_id}` 要求登录，仅允许访问本账号图片；前端带鉴权加载并使用临时 Blob URL 展示。
- 聊天历史保存图片引用，工具给模型的结果只包含简短说明，不包含 Base64 图像。
- API 超时或失败不自动重复生图，错误交给现有工具错误处理流程；日志保留诊断信息。
- 修改 `.env` 后需重启服务，以刷新模型实例和工具配置。生成图片可能产生单独的 API 费用。

## 验证

后端：`uv run python -m unittest tests.test_generated_images`。

前端：`node --experimental-strip-types --test src/lib/chat-images.test.ts src/lib/chat-stream.test.ts src/lib/chat-timeline.test.ts`，再运行 `npm run build`。

自动测试模拟外部 API，覆盖图片流式传递、历史恢复、账号隔离、失败处理和 GPT 请求格式。
