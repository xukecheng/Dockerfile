# stable-diffusion.cpp (sd-server, CUDA)

上游 https://github.com/leejet/stable-diffusion.cpp 的 `sd-server` CUDA 构建，固定 commit 见 Dockerfile 的 `SD_CPP_REF`。
默认只编译 sm_89（RTX 40 系），其它显卡改 `CUDA_ARCHITECTURES` build-arg。

镜像入口就是 `sd-server --listen-ip 0.0.0.0 --listen-port 1234`，模型参数通过容器命令追加。

## Qwen-Image-2.1 用法（Unraid 10.10.10.2）

模型放 `/mnt/user/appdata/sdcpp/models/`，文件清单见上游文档
https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md

```bash
docker run -d --name sdcpp --gpus all --restart unless-stopped \
  -p 1234:1234 \
  -v /mnt/user/appdata/sdcpp/models:/models \
  -v /mnt/user/appdata/sdcpp/output:/output \
  ghcr.io/xukecheng/stable-diffusion.cpp:latest \
  --diffusion-model /models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors \
  --vae /models/vae/qwen_image_2.1_vae_bf16.safetensors \
  --llm /models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --llm_vision /models/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  --diffusion-fa --offload-to-cpu --cfg-scale 6.0 --sampling-method euler
```

- Web UI: `http://10.10.10.2:1234/`
- OpenAI 兼容: `POST /v1/images/generations`
- A1111 兼容: `/sdapi/v1/txt2img`
- 原生异步 API: `/sdcpp/v1/...`
