FROM huggingface/transformers-pytorch-gpu:4.41.2

WORKDIR /app
ADD ./openai_api_server.py openai_api_server.py
ADD ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV PORT 8000
ENV MODEL_PATH THUDM/glm-4-9b-chat

CMD ["python", "openai_api_server.py"]