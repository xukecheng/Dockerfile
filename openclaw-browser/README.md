# OpenClaw Sandbox Browser

为 [OpenClaw](https://github.com/anthropics/claude-code) 定制的 Chromium 浏览器 sidecar 容器，通过 CDP (Chrome DevTools Protocol) 提供远程浏览器自动化能力。

## 核心特性

- **CDP 反向代理**: Caddy 反代绕过 Chrome M113+ 的 `--remote-debugging-address=0.0.0.0` 安全限制
- **中文支持**: zh_CN.UTF-8 locale + Noto CJK 字体
- **noVNC**: 可选的 Web 远程桌面，方便调试观察
- **GPU 加速**: 自动检测 `/dev/dri` 设备并启用硬件加速
- **反检测**: 禁用 `AutomationControlled` 特征，正常 User-Agent
- **崩溃恢复**: Chromium 崩溃后自动清理锁文件并重启
- **持久化**: Chrome profile 可挂载持久化，插件和 cookie 重启不丢失

## 快速开始

```bash
# 构建
docker build -t openclaw-browser:local .

# 运行
docker run -d --name openclaw-browser \
  --shm-size=1g \
  -p 6080:6080 \
  -p 9222:9222 \
  openclaw-browser:local

# 验证 CDP
curl http://localhost:9222/json/version

# 验证 noVNC — 浏览器打开 http://localhost:6080

# 验证中文 locale
docker exec openclaw-browser locale
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CDP_PORT` | `9222` | 对外暴露的 CDP 端口 |
| `VNC_PORT` | `5900` | VNC 端口 |
| `NOVNC_PORT` | `6080` | noVNC Web 端口 |
| `ENABLE_NOVNC` | `1` | 是否启用 noVNC（`0` 禁用） |
| `HEADLESS` | `0` | 是否 headless 模式（`1` 启用） |
| `CHROME_EXTRA_ARGS` | 空 | 用户自定义额外 Chrome 参数 |
| `SCREEN_RESOLUTION` | `1920x1080x24` | Xvfb 虚拟屏幕分辨率 |

## Docker Compose 示例

与 OpenClaw Gateway 配合使用：

```yaml
services:
  openclaw-browser:
    image: ghcr.io/xukecheng/openclaw-browser:latest
    container_name: openclaw-browser
    shm_size: "1g"
    ports:
      - "6080:6080"   # noVNC
      - "9222:9222"   # CDP
    environment:
      - ENABLE_NOVNC=1
      - HEADLESS=0
    volumes:
      - ./browser-data:/data/chrome-profile
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
| **Network Type** | 与 OpenClaw 同一网络（或 bridge） |
| **Extra Parameters** | `--shm-size=1g` |
| **Port: 6080** | noVNC Web 界面 |
| **Port: 9222** | CDP 协议端口 |
| **Volume** | `/mnt/user/appdata/openclaw-browser` -> `/data/chrome-profile` |
| **Device (可选)** | `/dev/dri/card1`, `/dev/dri/renderD129` |

环境变量按需添加，参考上方环境变量表。

## GPU 加速

容器启动时自动检测 `/dev/dri/renderD*` 设备。如果存在，会自动添加 GPU 相关 Chromium 参数。

需要在 Docker 运行时映射 GPU 设备：

```bash
docker run -d --shm-size=1g \
  --device /dev/dri/card1:/dev/dri/card1 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -p 9222:9222 -p 6080:6080 \
  openclaw-browser:local
```

## 技术原理

Chrome M113+ 在源码层面强制将 `--remote-debugging-address=0.0.0.0` 覆写为 `127.0.0.1`（Chromium Issue 1425667, WontFix）。

解决方案：容器内 Caddy 反向代理监听 `0.0.0.0:<CDP_PORT>`，转发到 `127.0.0.1:<内部端口>`，同时改写 `Host` header 为 `127.0.0.1`，绕过 Chrome 的安全检查。

## 致谢

架构参考 [canyugs/openclaw-sandbox-browser](https://github.com/canyugs/openclaw-sandbox-browser)（MIT License）。

## License

MIT
