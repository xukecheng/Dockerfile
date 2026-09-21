# comfyui-mcp

把 Unraid 上的 ComfyUI（Qwen-Image-2.1）包成 MCP 服务，streamable-http，路径 `/mcp`，端口 8189。
所有调用方共用 ComfyUI 的单 GPU FIFO 队列；等待期间通过 MCP progress 汇报进度；队列超过 `MAX_PENDING` 直接拒绝。

工具：`generate_image(prompt, aspect, quality, seed?, caller?)`、`edit_image(image, prompt, quality, seed?, caller?)`、`queue_status()`。
提示词写法和 draft/final 的选择规则都写在服务的 `instructions` 和工具描述里，客户端接上即得，不需要额外 skill 文档。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `COMFY_URL` | `http://comfyui:8188` | 容器内访问 ComfyUI（同一 docker 网络用容器名） |
| `COMFY_PUBLIC_URL` | `http://10.10.10.2:8188` | 返回给调用方的图片 URL 前缀 |
| `OUTPUT_HOST_DIR` | `/mnt/user/appdata/comfyui/basedir/output` | 返回给调用方的宿主机路径前缀 |
| `MAX_PENDING` | `5` | 排队上限 |
| `PORT` | `8189` | |

## 接入

```bash
claude mcp add --transport http qwen-image http://10.10.10.2:8189/mcp
```

Hermes：`config.yaml` 里加一条 HTTP 类型的 MCP server，`url: http://10.10.10.2:8189/mcp`。
