ARG BASE_TAG="1.16.1"
ARG BASE_IMAGE="chrome"
FROM kasmweb/${BASE_IMAGE}:${BASE_TAG}

LABEL org.opencontainers.image.source="https://github.com/xukecheng/Dockerfile"
LABEL org.opencontainers.image.description="Customized Chrome browser sandbox for OpenClaw with CDP reverse proxy"
LABEL org.opencontainers.image.licenses="MIT"

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Remove expired Google Chrome apt source from base image
RUN rm -f /etc/apt/sources.list.d/google-chrome.list 2>/dev/null

# Add Caddy official apt repo + install packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends debian-keyring debian-archive-keyring apt-transport-https curl \
  && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
  && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list \
  && apt-get update \
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
ENV VNC_USER=xukecheng

# GPU: KasmVNC VirtualGL device paths (override at runtime to match --device mapping)
ENV KASM_EGL_CARD=/dev/dri/card0
ENV KASM_RENDERD=/dev/dri/renderD128

# Fix VirtualGL: base image wrapper checks -O (file owned by user) which fails
# for /dev/dri/* devices owned by root. Replace with -r (readable) check.
RUN sed -i 's/ -O / -r /g' /usr/bin/google-chrome

# KasmVNC quality (env vars override yaml config at runtime)
ENV KASMVNC_DYNAMIC_QUALITY_MIN=9 \
    KASMVNC_DYNAMIC_QUALITY_MAX=9 \
    KASMVNC_TREAT_LOSSLESS=9 \
    KASMVNC_MAX_FRAME_RATE=60

# Replace custom_startup.sh to add Caddy CDP proxy
COPY custom_startup.sh /dockerstartup/custom_startup.sh
RUN chmod +x /dockerstartup/custom_startup.sh

EXPOSE 9222

USER 1000
