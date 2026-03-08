FROM debian:bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/xukecheng/Dockerfile"
LABEL org.opencontainers.image.description="Customized Chromium browser sandbox for OpenClaw with CDP reverse proxy"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

# Install packages: chromium, caddy, xvfb, vnc, fonts, locales
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    caddy \
    chromium \
    curl \
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-noto-cjk \
    jq \
    locales \
    novnc \
    procps \
    websockify \
    x11vnc \
    xvfb \
  && rm -rf /var/lib/apt/lists/*

# Generate zh_CN.UTF-8 locale
RUN sed -i 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen

ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    TZ=Asia/Shanghai

# Chrome profile mount point
RUN mkdir -p /data/chrome-profile

COPY entrypoint.sh /usr/local/bin/openclaw-sandbox-browser
RUN chmod +x /usr/local/bin/openclaw-sandbox-browser

EXPOSE 9222 5900 6080

CMD ["openclaw-sandbox-browser"]
