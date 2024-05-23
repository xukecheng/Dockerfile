# 使用 debian:stable 作为基础镜像
FROM debian:stable

# 设置环境变量以避免 tzdata 的交互提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装必要的系统依赖和 Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    openssh-server \
    sudo \
    vim \
    git \
    git-lfs \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 3.12 的软链接
RUN ln -s /usr/bin/python3.12 /usr/bin/python3

# 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

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

# 启动 SSH 服务
CMD ["/usr/sbin/sshd", "-D"]
