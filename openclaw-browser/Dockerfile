ARG BASE_TAG="1.16.1"
ARG BASE_IMAGE="chrome"
FROM kasmweb/${BASE_IMAGE}:${BASE_TAG}

LABEL org.opencontainers.image.source="https://github.com/xukecheng/Dockerfile"
LABEL org.opencontainers.image.description="Customized Chrome browser sandbox for OpenClaw with CDP reverse proxy"
LABEL org.opencontainers.image.licenses="MIT"

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Install Caddy (CDP reverse proxy), locales, CJK fonts, GPU drivers
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    caddy \
    fonts-noto-cjk \
    locales \
    mesa-va-drivers \
  && rm -rf /var/lib/apt/lists/*

# Generate zh_CN.UTF-8 locale
RUN sed -i 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen

ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    TZ=Asia/Shanghai

# Chrome args: CDP + anti-detection (injected via APP_ARGS -> custom_startup.sh)
ENV APP_ARGS="--start-maximized --remote-debugging-address=127.0.0.1 --remote-debugging-port=9223 --disable-blink-features=AutomationControlled"
ENV CDP_PORT=9222

# Replace custom_startup.sh to add Caddy CDP proxy
COPY custom_startup.sh /dockerstartup/custom_startup.sh
RUN chmod +x /dockerstartup/custom_startup.sh

EXPOSE 9222

USER 1000
