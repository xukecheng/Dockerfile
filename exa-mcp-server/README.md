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

| 原工具名         | 新工具名 | 输入参数        | 描述                                             |
| ---------------- | -------- | --------------- | ------------------------------------------------ |
| `web_search_exa` | `search` | `query: string` | 网络搜索工具（固定返回5个结果）                  |
| `crawling_exa`   | `fetch`  | `id: string`    | 从搜索结果获取文档内容的工具（固定提取5000字符） |

### 输入模式 (Input Schema)

#### search 工具
```json
{
  "query": {
    "type": "string",
    "description": "Search query"
  }
}
```

#### fetch 工具  
```json
{
  "id": {
    "type": "string", 
    "description": "ID from search results to fetch document content"
  }
}
```

### 默认配置

- **search 工具**：固定返回 5 个搜索结果
- **fetch 工具**：固定提取 5000 个字符的文档内容

### 深度研究模型集成

根据深度研究模型的要求：
1. **search 工具**：接收查询并返回搜索结果
2. **fetch 工具**：接收搜索结果中的 `id` 并返回对应的文档

这确保了与深度研究模型的正确集成，简化了参数结构，避免了参数不匹配的问题。

### 修改的文件

1. **src/index.ts** - 更新工具注册表和工具注册逻辑
2. **src/tools/webSearch.ts** - 更新工具名称、请求 ID 和请求结构
3. **src/tools/crawling.ts** - 更新工具名称、请求 ID、参数和请求结构
4. **src/types.ts** - 简化 ExaSearchRequest 和 SearchArgs 接口

### TypeScript 接口变更

#### ExaSearchRequest 接口简化
```typescript
// 原接口（复杂）
export interface ExaSearchRequest {
  query: string;
  type: string;
  numResults: number;
  contents: { /* 复杂的嵌套结构 */ };
  // 其他可选字段...
}

// 新接口（简化）
export interface ExaSearchRequest {
  query: string;
}
```

#### SearchArgs 接口简化
```typescript
// 原接口
export interface SearchArgs {
  query: string;
  numResults?: number;
  livecrawl?: 'always' | 'fallback' | 'preferred';
}

// 新接口
export interface SearchArgs {
  query: string;
}
```

## 环境变量

- `EXA_API_KEY`: 必需，你的 Exa AI API 密钥

## 端口

- 容器暴露端口：3000

## 注意事项

- 确保你有有效的 Exa AI API 密钥
- 构建过程需要网络连接来克隆 GitHub 仓库
- 镜像基于 Node.js 18 Alpine Linux
