# 使用 Python 3.11.2 作为基础镜像
FROM python:3.11.2

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
    && rm -rf /var/lib/apt/lists/*

# 安装 pip 包并添加调试信息
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir \
    transformers \
    pandas \
    scikit-learn \
    torch==2.2.0 \
    coremltools \
    accelerate \
    imbalanced-learn \
    scikit-learn \
    accelerate \
    imblearn \
    wandb \
    ipykernel \
    jupyter \
    && echo "Installed packages:" \
    && pip list

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
