# 使用 PyTorch 官方镜像
FROM pytorch/pytorch

# 安装必要的系统依赖和 Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    openssh-server \
    sudo \
    vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 3.12 的软链接
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2

# 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 安装并配置 SSH
RUN mkdir /var/run/sshd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -i 's/ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config

# 设置公钥认证
RUN mkdir -p /root/.ssh
ARG SSH_PUBLIC_KEY
RUN echo "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys
RUN chown -R root:root /root/.ssh

# 暴露 SSH 端口
EXPOSE 22

# 启动 SSH 服务
CMD ["/usr/sbin/sshd", "-D"]
