# ComfyUI（固定版本）+ PyTorch cu130，用于 Unraid RTX 4060 Ti 跑 Qwen-Image-2.1
# torch 轮子自带 CUDA 运行库，宿主只需 nvidia 驱动 + nvidia container runtime，所以底包用 python-slim 而不是 nvidia/cuda
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ComfyUI README：20 系及以上必须用 cu130+；torch 版本尽量新但避开发布不足两周的
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.13.0
RUN pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url ${TORCH_INDEX}

ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_REF=v0.37.0
WORKDIR /comfyui
RUN git clone --depth 1 --branch ${COMFYUI_REF} ${COMFYUI_REPO} . && \
    pip install -r requirements.txt

# 所有可变数据（models/input/output/custom_nodes/user）都放 /basedir，由 --base-directory 指向
VOLUME ["/basedir"]
EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--base-directory", "/basedir"]
