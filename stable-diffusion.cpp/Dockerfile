# stable-diffusion.cpp (sd-server) CUDA 镜像
# 基于上游 docker/Dockerfile.cuda 改写：源码在构建阶段 clone（固定 commit），
# 默认只编 RTX 4060 Ti (Ada, sm_89) 一种架构以缩短 CI 时间，入口改为 sd-server。
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build

# sd-server 构建时会把 Web UI 打进二进制，所以需要 Node/pnpm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ccache cmake ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    npm install -g pnpm@10.15.1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG SD_CPP_REPO=https://github.com/leejet/stable-diffusion.cpp.git
ARG SD_CPP_REF=c678dfe704a2230342376b46add9c8ca736a653d
WORKDIR /sd.cpp
RUN git clone --filter=blob:none ${SD_CPP_REPO} . && \
    git checkout ${SD_CPP_REF} && \
    git submodule update --init --recursive --depth 1

ARG CUDACXX=/usr/local/cuda/bin/nvcc
# 89 = Ada (RTX 40 系)。留空则编全部架构，构建时间会长很多
ARG CUDA_ARCHITECTURES=89
RUN cmake . -B ./build \
        -DSD_CUDA=ON \
        -DSD_BUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF \
        -DSD_BUILD_SHARED_GGML_LIB=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        ${CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}} && \
    cmake --build ./build --config Release -j"$(nproc)"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends libgomp1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build /sd.cpp/build/bin /sd.cpp/bin
RUN printf '#!/bin/sh\nexec /sd.cpp/bin/sd-cli "$@"\n' > /sd-cli && \
    printf '#!/bin/sh\nexec /sd.cpp/bin/sd-server "$@"\n' > /sd-server && \
    chmod +x /sd-cli /sd-server

VOLUME ["/models", "/output"]
EXPOSE 1234
ENTRYPOINT ["/sd-server", "--listen-ip", "0.0.0.0", "--listen-port", "1234"]
