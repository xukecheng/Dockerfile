# OpenClaw Sandbox Browser

基于 [kasmweb/chrome](https://hub.docker.com/r/kasmweb/chrome) 的 OpenClaw 浏览器 sidecar 容器，通过 CDP (Chrome DevTools Protocol) 提供远程浏览器自动化能力。

## 核心特性

- **KasmVNC 桌面**: 浏览器打开 `https://<host>:6901` 即可看到 Chrome 桌面，支持剪贴板同步、文件传输
- **CDP 反向代理**: Caddy 反代绕过 Chrome M113+ 的 CDP 安全限制，端口 9222 对外暴露
- **MCP Server**: 内置 [chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp)，通过 9222 端口 `/mcp` 路径提供 MCP 服务（截图、导航、点击、JS 执行等 37 个工具）
- **中文支持**: zh_CN.UTF-8 locale + Noto CJK 字体
- **反检测**: 禁用 `AutomationControlled` 特征
- **崩溃恢复**: kasmweb 内置进程守护，Chrome 崩溃自动重启
- **GPU 加速**: DRI3 直接渲染，Chrome ANGLE/EGL 直通 GPU（绕过 VirtualGL）

## 快速开始

```bash
# 构建
docker build -t openclaw-browser:local .

# 运行 (带 GPU 加速)
docker run -d --name openclaw-browser \
  --shm-size=1g \
  --group-add video \
  --device /dev/dri \
  -p 6901:6901 \
  -p 9222:9222 \
  -e VNC_PW=password \
  openclaw-browser:local

# 验证 CDP
curl http://localhost:9222/json/version

# 验证 MCP
curl http://localhost:9222/ping

# 验证 KasmVNC 桌面 — 浏览器打开 https://localhost:6901
# 默认密码: password
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VNC_PW` | `password` | KasmVNC 登录密码 |
| `VNC_USER` | `openclaw` | KasmVNC 登录用户名（可自定义） |
| `CDP_PORT` | `9222` | 对外暴露的 CDP 端口 |
| `APP_ARGS` | (见 Dockerfile) | Chrome 额外启动参数 |
| `LAUNCH_URL` | 空 | Chrome 启动时打开的 URL |
| `HW3D` | `true` | 启用 KasmVNC DRI3 直接渲染 |
| `DRINODE` | `/dev/dri/renderD128` | DRI3 使用的 GPU render 设备节点 |
| `KASMVNC_DYNAMIC_QUALITY_MIN` | `9` | KasmVNC 最低画质 (1-9) |
| `KASMVNC_DYNAMIC_QUALITY_MAX` | `9` | KasmVNC 最高画质 (1-9) |
| `KASMVNC_TREAT_LOSSLESS` | `9` | 无损渲染阈值 |
| `KASMVNC_MAX_FRAME_RATE` | `60` | 最大帧率 |

## Docker Compose 示例

与 OpenClaw Gateway 配合使用：

```yaml
services:
  openclaw-browser:
    image: ghcr.io/xukecheng/openclaw-browser:latest
    container_name: openclaw-browser
    shm_size: "1g"
    group_add:
      - video
    ports:
      - "6901:6901"   # KasmVNC Web 桌面
      - "9222:9222"   # CDP
    environment:
      - VNC_PW=password
      - DRINODE=/dev/dri/renderD128        # 按实际 GPU 设备调整
    volumes:
      - ./browser-data:/home/kasm-user
    devices:
      - /dev/dri:/dev/dri                 # GPU 加速
    networks:
      - openclaw-net

  openclaw-gateway:
    image: openclaw:local
    container_name: openclaw-gateway
    # ... OpenClaw 配置 ...
    networks:
      - openclaw-net

networks:
  openclaw-net:
    driver: bridge
```

OpenClaw `openclaw.json` 浏览器配置：

```json
{
  "browser": {
    "enabled": true,
    "attachOnly": true,
    "defaultProfile": "remote",
    "profiles": {
      "remote": {
        "cdpUrl": "http://openclaw-browser:9222"
      }
    }
  }
}
```

## Unraid 部署

在 Unraid Docker UI 中添加容器：

| 配置项 | 值 |
|---|---|
| **Repository** | `ghcr.io/xukecheng/openclaw-browser:latest` |
| **Network Type** | bridge |
| **Extra Parameters** | `--shm-size=1g --group-add video` |
| **Port: 6901** | KasmVNC Web 桌面 (HTTPS) |
| **Port: 9222** | CDP 协议端口 |
| **Volume** | `/mnt/user/appdata/openclaw-browser` -> `/home/kasm-user` |
| **Device** | `/dev/dri` (映射整个 GPU 目录) |
| **DRINODE** | `/dev/dri/renderD128` (按 `ls /dev/dri/` 确认) |
| **VNC_PW** | 设置 VNC 密码 |

## 技术原理

### CDP 反向代理

Chrome M113+ 在源码层面强制将 `--remote-debugging-address=0.0.0.0` 覆写为 `127.0.0.1`。

解决方案：容器内 Caddy 反向代理监听 `0.0.0.0:9222`，转发到 `127.0.0.1:9223`，同时改写 `Host` header 为 `127.0.0.1`，绕过 Chrome 的安全检查。

### GPU 加速 (DRI3)

Chrome 131+ 仅支持 ANGLE/EGL 渲染，不再支持 GLX。kasmweb 内置的 VirtualGL 拦截的是 GLX 调用，与 Chrome ANGLE/EGL 不兼容（VirtualGL 维护者已确认）。

解决方案：使用 KasmVNC DRI3 模式（`HW3D=true` + `DRINODE`），让 Chrome 通过 DRI3 协议直接访问 GPU，完全绕过 VirtualGL。注意不能设置 `KASM_EGL_CARD`/`KASM_RENDERD`，否则会触发 VirtualGL 反而干扰渲染。

### MCP Server

容器内置 [chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) + [mcp-proxy](https://github.com/punkpeye/mcp-proxy)，通过 Caddy 路径分流复用 9222 端口：

| 路径 | 后端 | 说明 |
|---|---|---|
| `/mcp` | mcp-proxy | MCP Streamable HTTP 端点 |
| `/sse` | mcp-proxy | MCP SSE 端点 |
| `/ping` | mcp-proxy | 健康检查 |
| 其余路径 | Chrome CDP | CDP 协议（不受影响） |

chrome-devtools-mcp 使用懒连接，首次工具调用时才连接 Chrome CDP，Chrome 重启后自动重连。

#### Claude Desktop 客户端配置

`~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "openclaw-browser": {
      "command": "npx",
      "args": ["mcp-remote", "http://<host>:9222/mcp"]
    }
  }
}
```

## 致谢

- [kasmweb/chrome](https://hub.docker.com/r/kasmweb/chrome) — 基础镜像
- [canyugs/openclaw-sandbox-browser](https://github.com/canyugs/openclaw-sandbox-browser) — Caddy CDP 反代方案

## License

MIT
