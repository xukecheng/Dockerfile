FROM alpine:3.9 as downloader
RUN apk update && apk add --no-cache wget
RUN wget https://github.com/Torantulino/Auto-GPT/archive/refs/tags/v0.1.2.tar.gz \
  && tar -zxvf v0.1.2.tar.gz

FROM python:3.11-slim as runner
ENV PIP_NO_CACHE_DIR=yes
WORKDIR /app
COPY --from=downloader /Auto-GPT-0.1.2/scripts/ .
COPY --from=downloader /Auto-GPT-0.1.2/requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]