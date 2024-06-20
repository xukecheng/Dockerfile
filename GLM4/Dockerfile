FROM python:3.11.2

WORKDIR /app
ADD ./openai_api_server.py openai_api_server.py
ADD ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV MODEL_PATH THUDM/glm-4-9b-chat
ENV HF_HOME /models

CMD ["python", "/app/openai_api_server.py"]