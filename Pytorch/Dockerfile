# 使用 debian:stable 作为基础镜像
FROM debian:stable

# 设置环境变量以避免 tzdata 的交互提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    openssh-server \
    sudo \
    vim \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz \
    && tar -xzf Python-3.12.0.tgz \
    && cd Python-3.12.0 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.12 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && cd .. \
    && rm -rf Python-3.12.0 Python-3.12.0.tgz

# 安装 pip 包并添加调试信息
RUN python3.12 -m pip install --no-cache-dir -U pip \
    && python3.12 -m pip install --no-cache-dir \
    transformers \
    pandas \
    scikit-learn \
    torch \
    accelerate \
    imbalanced-learn \
    wandb \
    ipykernel \
    jupyter \
    && echo "Installed packages:" \
    && python3.12 -m pip list

# 安装并配置 SSH
RUN mkdir /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config

# 设置公钥认证
RUN mkdir -p /root/.ssh
ARG SSH_PUBLIC_KEY
RUN echo "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && chown -R root:root /root/.ssh

# 暴露 SSH 端口
EXPOSE 22

# 启动 SSH 服务
CMD ["/usr/sbin/sshd", "-D"]
