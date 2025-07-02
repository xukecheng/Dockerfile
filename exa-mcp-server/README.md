# Exa MCP Server Docker 构建

本项目的 Dockerfile 用于自动克隆 exa-mcp-server 仓库并批量重命名、精简工具接口，适配深度研究模型。

## 主要特性
- `web_search_exa` 工具重命名为 `search`，参数仅保留 `query: string`
- `crawling_exa` 工具重命名为 `fetch`，参数仅保留 `id: string`
- TypeScript 类型定义、工具注册、请求结构等全部自动批量替换
- 完全兼容深度研究模型的 search/fetch 工具调用规范

## 构建与运行

```bash
docker run -e EXA_API_KEY=your_actual_api_key -p 8181:8181 exa-mcp-server:latest
```

## 工具接口说明

| 工具名 | 输入参数      |
| ------ | ------------- |
| search | query: string |
| fetch  | id: string    |

### search 工具 input schema
```json
{
  "query": {
    "type": "string",
    "description": "Search query"
  }
}
```

### fetch 工具 input schema
```json
{
  "id": {
    "type": "string",
    "description": "ID from search results to fetch document content"
  }
}
```

## 深度研究模型集成
- search 工具：接收查询并返回搜索结果
- fetch 工具：接收 search 结果中的 id 并返回文档内容
- 参数结构极简，完全避免敏感信息外泄风险

## 主要自动修改的文件
- src/index.ts
- src/tools/webSearch.ts
- src/tools/crawling.ts
- src/types.ts

## TypeScript 类型变更示例
```typescript
// ExaSearchRequest
export interface ExaSearchRequest {
  query: string;
}
// SearchArgs
export interface SearchArgs {
  query: string;
}
```

## 端口与环境变量
- 端口：8181
- 环境变量：EXA_API_KEY（必填）