FROM python:3.12-slim
WORKDIR /app

# 安装必要的依赖
RUN pip install mcpo uv

# 创建配置文件目录
RUN mkdir -p /app/config

# 将配置文件添加到容器中
COPY mcp-config.json /app/config/

# 端口定义
EXPOSE 11436

# 使用配置文件启动mcpo
CMD ["uvx", "mcpo", "--host", "0.0.0.0", "--port", "11436", "--config", "/app/config/mcp-config.json"]