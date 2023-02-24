FROM python:3.9.16-alpine3.17
WORKDIR /app
COPY main.py .
RUN pip install bs4 requests
CMD ["python", "/app/main.py"]
