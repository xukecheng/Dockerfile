# Exa MCP Server Docker 构建

这个 Dockerfile 用于构建一个修改版本的 exa-mcp-server，其中：

- `web_search_exa` 工具名称改为 `search`
- `crawling_exa` 工具名称改为 `fetch`

## 构建步骤

1. **克隆仓库并修改代码**：Dockerfile 会自动从 GitHub 克隆 exa-mcp-server 仓库
2. **修改工具名称**：使用 sed 命令批量替换工具名称
3. **构建应用**：安装依赖并构建 TypeScript 代码
4. **创建运行镜像**：使用多阶段构建创建精简的运行镜像

## 使用方法

### 构建镜像

```bash
# 使用提供的构建脚本
chmod +x build.sh
./build.sh

# 或直接使用 docker build
docker build -t exa-mcp-server:latest .
```

### 运行容器

```bash
# 设置你的 Exa API 密钥并运行
docker run -e EXA_API_KEY=your_actual_api_key -p 3000:3000 exa-mcp-server:latest

# 或者交互式运行
docker run -it -e EXA_API_KEY=your_actual_api_key -p 3000:3000 exa-mcp-server:latest
```

## 修改内容

### 工具名称变更

| 原工具名         | 新工具名 | 描述             |
| ---------------- | -------- | ---------------- |
| `web_search_exa` | `search` | 网络搜索工具     |
| `crawling_exa`   | `fetch`  | 网页内容抓取工具 |

### 修改的文件

1. **src/index.ts** - 更新工具注册表和工具注册逻辑
2. **src/tools/webSearch.ts** - 更新工具名称和请求 ID
3. **src/tools/crawling.ts** - 更新工具名称和请求 ID

## 环境变量

- `EXA_API_KEY`: 必需，你的 Exa AI API 密钥

## 端口

- 容器暴露端口：3000

## 注意事项

- 确保你有有效的 Exa AI API 密钥
- 构建过程需要网络连接来克隆 GitHub 仓库
- 镜像基于 Node.js 18 Alpine Linux
