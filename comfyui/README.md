# ComfyUI (CUDA, 固定版本)

`comfyanonymous/ComfyUI` 固定 tag 构建（见 Dockerfile `COMFYUI_REF`），PyTorch cu130。
不带 ComfyUI-Manager，自定义节点放 `/basedir/custom_nodes`。

## Unraid 用法（10.10.10.2）

```bash
docker run -d --name comfyui --runtime nvidia --gpus all --restart unless-stopped \
  -p 8188:8188 \
  -v /mnt/user/appdata/comfyui/basedir:/basedir \
  ghcr.io/xukecheng/comfyui:latest
```

模型目录结构（ComfyUI 标准）：

```
/basedir/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors   # Comfy-Org/Qwen-Image-2.1
/basedir/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors          # Comfy-Org/Qwen-Image-2.1
/basedir/models/vae/qwen_image_2.1_vae_bf16.safetensors                     # Comfy-Org/Qwen-Image-2.1
```

- Web UI: `http://10.10.10.2:8188/`
- API: `POST /prompt`（工作流 JSON，API 格式）→ `GET /history/{id}` → `GET /view?filename=…`
- 升级：改 `COMFYUI_REF` / `TORCH_VERSION` 后 push，CI 自动构建
