# comfyui-mcp：把 ComfyUI 上的 Qwen-Image-2.1 暴露成 MCP（streamable-http :8189）
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py .
EXPOSE 8189
ENTRYPOINT ["python", "server.py"]
