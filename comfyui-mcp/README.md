# comfyui-mcp

把 Unraid 上的 ComfyUI（Qwen-Image-2.1）包成 MCP 服务，streamable-http，路径 `/mcp`，端口 8189。
所有调用方共用 ComfyUI 的单 GPU FIFO 队列；等待期间通过 MCP progress 汇报进度；队列超过 `MAX_PENDING` 直接拒绝。

工具：`generate_image(prompt, aspect, seed?)`、`edit_image(image, prompt, seed?)`、`request_upload()`（本机图片上传：签发 10 分钟一次性签名 URL，客户端 `curl -F image=@file`）、`queue_status()`。
一律 1K、40 步（文生图 1 分钟、编辑 1.5 分钟），不提供 2K。返回全分辨率 JPEG（Claude API 单图 5 MB 上限，2K PNG 传不过）+ PNG 原图签名 URL；不暴露服务器路径。调用方断开时只撤销排队中的任务，正在跑的让它跑完。
提示词写法和 draft/final 的选择规则都写在服务的 `instructions` 和工具描述里，客户端接上即得，不需要额外 skill 文档。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `COMFY_URL` | `http://comfyui:8188` | 容器内访问 ComfyUI（同一 docker 网络用容器名） |
| `PUBLIC_URL` | `http://10.10.10.2:8189` | 本服务对调用方的地址，图片 URL 前缀（走 Cloudflare 时改成公网域名） |
| `MCP_AUTH_TOKENS` | 空 | `name1:token1,name2:token2`。非空即启用 Bearer 鉴权，`name` 会作为 caller 写进输出文件名；为空则不鉴权，只能在纯内网用 |
| `IMAGE_SIGN_KEY` | 取第一个 token | 图片 URL 的 HMAC 签名密钥 |
| `MAX_PENDING` | `5` | 排队上限 |
| `PORT` | `8189` | |

## 安全

- MCP 端点 `/mcp`：Bearer token（`MCP_AUTH_TOKENS`），401 带标准 `WWW-Authenticate`
- 图片 `/image/<sub>/<file>?sig=`：HMAC 签名 URL，无需 header，篡改即 403；由本服务代理 ComfyUI 的 `/view`，**ComfyUI 8188 永远不要对外暴露**（`/prompt` 能执行任意工作流）
- 上传 `POST /upload?id&exp&sig`：一次性签名 + 10 分钟有效 + 30 MB 上限 + 必须是图片
- `/health`：无鉴权，只返回队列状态

对外发布走 Cloudflare Tunnel + Access（Service Token）；应用层 token 作为第二道。

## 接入

```bash
claude mcp add --transport http qwen-image https://<host>/mcp --header "Authorization: Bearer <token>"
```

Hermes：`config.yaml` 的 `mcp_servers` 加一条 `url` + `headers: {Authorization: Bearer <token>}`，`timeout` 给到 1800（2K 一张 4 分钟，加排队）。
