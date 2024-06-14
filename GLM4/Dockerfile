FROM python:3.11.2

WORKDIR /app
ADD ./openai_api_server.py openai_api_server.py
ADD ./requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT 8000
ENV MODEL_PATH THUDM/glm-4-9b-chat

CMD ["python", "/app/openai_api_server.py"]