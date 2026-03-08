# OpenClaw Sandbox Browser

基于 [kasmweb/chrome](https://hub.docker.com/r/kasmweb/chrome) 的 OpenClaw 浏览器 sidecar 容器，通过 CDP (Chrome DevTools Protocol) 提供远程浏览器自动化能力。

## 核心特性

- **KasmVNC 桌面**: 浏览器打开 `https://<host>:6901` 即可看到 Chrome 桌面，支持剪贴板同步、文件传输
- **CDP 反向代理**: Caddy 反代绕过 Chrome M113+ 的 CDP 安全限制，端口 9222 对外暴露
- **中文支持**: zh_CN.UTF-8 locale + Noto CJK 字体
- **反检测**: 禁用 `AutomationControlled` 特征
- **崩溃恢复**: kasmweb 内置进程守护，Chrome 崩溃自动重启
- **GPU 加速**: kasmweb 内置 VirtualGL 支持

## 快速开始

```bash
# 构建
docker build -t openclaw-browser:local .

# 运行
docker run -d --name openclaw-browser \
  --shm-size=1g \
  -p 6901:6901 \
  -p 9222:9222 \
  -e VNC_PW=password \
  openclaw-browser:local

# 验证 CDP
curl http://localhost:9222/json/version

# 验证 KasmVNC 桌面 — 浏览器打开 https://localhost:6901
# 默认密码: password
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VNC_PW` | `password` | KasmVNC 登录密码 |
| `CDP_PORT` | `9222` | 对外暴露的 CDP 端口 |
| `APP_ARGS` | (见 Dockerfile) | Chrome 额外启动参数 |
| `LAUNCH_URL` | 空 | Chrome 启动时打开的 URL |

## Docker Compose 示例

与 OpenClaw Gateway 配合使用：

```yaml
services:
  openclaw-browser:
    image: ghcr.io/xukecheng/openclaw-browser:latest
    container_name: openclaw-browser
    shm_size: "1g"
    ports:
      - "6901:6901"   # KasmVNC Web 桌面
      - "9222:9222"   # CDP
    environment:
      - VNC_PW=password
    volumes:
      - ./browser-data:/home/kasm-user
    devices:
      - /dev/dri/card1:/dev/dri/card1              # Intel 核显（可选）
      - /dev/dri/renderD129:/dev/dri/renderD129
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
| **Extra Parameters** | `--shm-size=1g` |
| **Port: 6901** | KasmVNC Web 桌面 (HTTPS) |
| **Port: 9222** | CDP 协议端口 |
| **Volume** | `/mnt/user/appdata/openclaw-browser` -> `/home/kasm-user` |
| **Device (可选)** | `/dev/dri/card1`, `/dev/dri/renderD129` |
| **VNC_PW** | 设置 VNC 密码 |

## 技术原理

Chrome M113+ 在源码层面强制将 `--remote-debugging-address=0.0.0.0` 覆写为 `127.0.0.1`。

解决方案：容器内 Caddy 反向代理监听 `0.0.0.0:9222`，转发到 `127.0.0.1:9223`，同时改写 `Host` header 为 `127.0.0.1`，绕过 Chrome 的安全检查。

## 致谢

- [kasmweb/chrome](https://hub.docker.com/r/kasmweb/chrome) — 基础镜像
- [canyugs/openclaw-sandbox-browser](https://github.com/canyugs/openclaw-sandbox-browser) — Caddy CDP 反代方案

## License

MIT
