#!/usr/bin/env python3
"""comfyui-mcp — 把 ComfyUI 上的 Qwen-Image-2.1 暴露成 MCP（streamable-http）。

工具: generate_image / edit_image / queue_status
特点:
- 无状态；所有调用方共用 ComfyUI 的单 GPU FIFO 队列
- 等待期间通过 MCP progress 通知汇报排队位置和采样步数（客户端不会超时）
- 队列过长直接拒绝，防止 agent 循环提交
- 返回一张 ≤1024 的 JPEG 预览 + 原图在 Unraid 上的路径和 URL
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import websockets
from PIL import Image as PILImage

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")          # 容器内访问 ComfyUI
COMFY_PUBLIC_URL = os.environ.get("COMFY_PUBLIC_URL", "http://10.10.10.2:8188").rstrip("/")  # 返回给调用方的 URL
OUTPUT_HOST_DIR = os.environ.get("OUTPUT_HOST_DIR", "/mnt/user/appdata/comfyui/basedir/output")  # Unraid 宿主机路径
MAX_PENDING = int(os.environ.get("MAX_PENDING", "5"))
PORT = int(os.environ.get("PORT", "8189"))

UNET = os.environ.get("UNET_NAME", "qwen_image_2.1_int8_convrot.safetensors")
CLIP = os.environ.get("CLIP_NAME", "qwen3vl_8b_int8_convrot.safetensors")
VAE = os.environ.get("VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors")

# 官方 Qwen-Image-2.1 README 推荐的 2K 尺寸表（final）；draft 约为其一半、对齐 32
SIZES = {
    "1:1":  {"final": (2048, 2048), "draft": (1024, 1024)},
    "4:3":  {"final": (2400, 1792), "draft": (1216, 896)},
    "3:4":  {"final": (1792, 2400), "draft": (896, 1216)},
    "3:2":  {"final": (2528, 1696), "draft": (1280, 864)},
    "2:3":  {"final": (1696, 2528), "draft": (864, 1280)},
    "16:9": {"final": (2752, 1536), "draft": (1376, 768)},
    "9:16": {"final": (1536, 2752), "draft": (768, 1376)},
}
STEPS = {"final": 40, "draft": 20}
EDIT_STEPS = {"final": 40, "draft": 40}          # 编辑 20 步会过锐/HDR 感，draft 只降分辨率不降步数
EDIT_REF_RES = {"final": 2048, "draft": 1024}   # 编辑模式：参考图/画布的目标像素规模
# 实测（RTX 4060 Ti 16GB, int8）用于预估等待时间，单位秒
EST_SECONDS = {("generate", "final"): 240, ("generate", "draft"): 35, ("edit", "final"): 300, ("edit", "draft"): 90}

INSTRUCTIONS = """本服务在家里 Unraid 的 RTX 4060 Ti 上跑 Qwen-Image-2.1（文生图 + 图片编辑，官方 int8 重打包权重），通过 ComfyUI 执行。

何时用：用户要生成图片、海报、插画、示意图，或要修改一张已有图片（换物体、改风格、改文字、扩展画面）。

必须知道的限制：
- 单 GPU、单队列：一次只能出一张图，所有调用方（Claude Code / Hermes）共用同一条 FIFO 队列。final 档 2K 一张约 4 分钟，draft 约 35 秒。工具会阻塞到出图为止并持续汇报进度，不要重复提交同一请求。
- 队列里已有 >5 个任务时工具会直接拒绝并给出预估等待，此时告诉用户稍后再试，不要循环重试。

提示词写法（实测结论）：
- 图里要出现的文字必须用引号原样写出（中英文都一样），如：招牌上写着"老陈肠粉"。不写具体文字，模型会画出乱码笔画。
- 中文场景用中文写提示词，模型的文本编码器对中文是原生的。
- 说清楚、不要堆砌：具体描述每个区域是什么媒介/材质/内容（"上半是真实照片，下半是米白亚麻布上的贴布绣"），比堆一串风格形容词有效得多。
- 负面提示词在当前配置（cfg 1.0）下不生效，不要依赖它。
- 编辑模式会锚定参考图的构图和人物身份；一次只描述一个明确的改动最稳，多个改动请分多次调用串联。
"""

mcp = MCPServer("qwen-image", instructions=INSTRUCTIONS)


# ---------- ComfyUI 交互 ----------

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=COMFY_URL, timeout=120)


async def _queue_state() -> tuple[int, int]:
    async with _client() as c:
        q = (await c.get("/queue")).json()
    return len(q.get("queue_running", [])), len(q.get("queue_pending", []))


async def _queue_position(pid: str) -> int | None:
    """返回排队位置：0=正在执行，n=前面还有 n 个；None=不在队列里（已完成或失败）"""
    async with _client() as c:
        q = (await c.get("/queue")).json()
    for j in q.get("queue_running", []):
        if j[1] == pid:
            return 0
    for i, j in enumerate(q.get("queue_pending", [])):
        if j[1] == pid:
            return i + 1
    return None


def _base_graph(prompt: str, steps: int, seed: int, prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                                                    "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def t2i_graph(prompt: str, w: int, h: int, steps: int, seed: int, prefix: str) -> dict:
    g = _base_graph(prompt, steps, seed, prefix)
    g["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}}
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    return g


def edit_graph(prompt: str, image_name: str, ref_res: int, steps: int, seed: int, prefix: str) -> dict:
    g = _base_graph(prompt, steps, seed, prefix)
    g["9"] = {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}}
    g["10"] = {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}}
    # 注意：autogrow 输入在 API 里必须扁平写成 images.image_N，嵌套 dict 会被静默丢弃
    g["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                          "negative_prompt": "", "resolution": ref_res, "images.image_1": ["9", 0]}}
    g["6"]["inputs"]["model"] = ["10", 0]
    g["6"]["inputs"]["latent_image"] = ["4", 2]   # 画布跟随参考图
    return g


async def _load_image_bytes(image: str) -> bytes:
    """支持三种形式：http(s) URL / data URI 或裸 base64 / Unraid 宿主机路径（需挂载进容器）"""
    if image.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            r = await c.get(image)
            r.raise_for_status()
            return r.content
    if image.startswith("data:"):
        image = image.split(",", 1)[1]
    if os.path.exists(image):
        return Path(image).read_bytes()
    try:
        return base64.b64decode(image, validate=True)
    except Exception:
        raise ToolError("image 参数必须是 http(s) URL、base64（可带 data: 前缀）或容器可见的文件路径")


async def _upload(data: bytes) -> str:
    name = f"mcp_{uuid.uuid4().hex[:12]}.png"
    async with _client() as c:
        r = await c.post("/upload/image", files={"image": (name, data, "image/png")})
        r.raise_for_status()
    return r.json()["name"]


async def _run(graph: dict, kind: str, quality: str, ctx: Context) -> dict:
    """提交并等待，期间用 progress 汇报；返回 history 里的 image 描述"""
    running, pending = await _queue_state()
    if pending >= MAX_PENDING:
        est = (running + pending) * EST_SECONDS[(kind, quality)]
        raise ToolError(f"队列已有 {running} 个执行中 + {pending} 个排队，超过上限 {MAX_PENDING}。预计等待约 {est // 60} 分钟后再试。")

    client_id = uuid.uuid4().hex
    total_steps = graph["6"]["inputs"]["steps"]
    async with _client() as c:
        r = await c.post("/prompt", json={"prompt": graph, "client_id": client_id})
        body = r.json()
    if "prompt_id" not in body:
        raise ToolError(f"ComfyUI 拒绝了任务: {json.dumps(body, ensure_ascii=False)[:800]}")
    pid = body["prompt_id"]
    t0 = time.time()

    ws_url = COMFY_URL.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
    step = 0
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                except asyncio.TimeoutError:
                    raw = None
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    mt, d = msg.get("type"), msg.get("data", {})
                    if mt == "progress" and d.get("prompt_id") == pid:
                        step = d.get("value", 0)
                        await ctx.report_progress(step, total_steps, f"采样 {step}/{total_steps} 步 · 已用 {time.time() - t0:.0f}s")
                        continue
                    if mt == "execution_error" and d.get("prompt_id") == pid:
                        raise ToolError(f"ComfyUI 执行出错: {d.get('exception_message', '')[:600]}")
                    if mt == "execution_success" and d.get("prompt_id") == pid:
                        break
                pos = await _queue_position(pid)
                if pos is None:
                    break
                if pos > 0:
                    await ctx.report_progress(0, total_steps, f"排队中，前面还有 {pos} 个任务 · 已等 {time.time() - t0:.0f}s")
    except asyncio.CancelledError:
        # 调用方放弃等待：把自己的任务撤掉，别留孤儿任务占 GPU
        async with _client() as c:
            pos = await _queue_position(pid)
            if pos == 0:
                await c.post("/interrupt")
            elif pos:
                await c.post("/queue", json={"delete": [pid]})
        raise

    async with _client() as c:
        hist = (await c.get(f"/history/{pid}")).json()
    if pid not in hist:
        raise ToolError("任务结束但 history 里找不到结果")
    st = hist[pid].get("status", {})
    if st.get("status_str") == "error":
        raise ToolError(f"ComfyUI 执行出错: {json.dumps(st, ensure_ascii=False)[:800]}")
    outs = [o for o in hist[pid]["outputs"].values() if "images" in o]
    if not outs:
        raise ToolError("任务完成但没有图片输出")
    im = outs[0]["images"][0]
    im["elapsed"] = time.time() - t0
    return im


async def _fetch_result(im: dict) -> tuple[bytes, str, str]:
    """返回 (原图字节, 宿主机路径, 可访问 URL)"""
    sub = im.get("subfolder", "")
    q = f"filename={im['filename']}&subfolder={sub}&type={im['type']}"
    async with _client() as c:
        data = (await c.get(f"/view?{q}")).content
    host_path = os.path.join(OUTPUT_HOST_DIR, sub, im["filename"]) if sub else os.path.join(OUTPUT_HOST_DIR, im["filename"])
    return data, host_path, f"{COMFY_PUBLIC_URL}/view?{q}"


def _preview(data: bytes, max_side: int = 1024) -> tuple[Image, tuple[int, int]]:
    img = PILImage.open(io.BytesIO(data))
    full = img.size
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg"), full


def _prefix(kind: str, caller: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", caller)[:24] or "mcp"
    return f"mcp/{datetime.now():%Y%m%d}/{safe}_{kind}"


# ---------- 工具 ----------

@mcp.tool()
async def generate_image(
    prompt: str,
    ctx: Context,
    aspect: Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"] = "1:1",
    quality: Literal["final", "draft"] = "final",
    seed: int | None = None,
    caller: str = "unknown",
) -> list:
    """用 Qwen-Image-2.1 文生图。阻塞直到出图（final 约 4 分钟，draft 约 35 秒），期间会持续汇报进度。

    quality 怎么选：
    - final（默认）：官方 2K 分辨率、40 步。用户要一张能用的图、没有说要快 → 用这个。
    - draft：约 1024 级、20 步。用户说"先看看 / 快速试试 / 出几个方向 / 草稿"，或同一提示词要连续改多版 → 用这个；满意后再用 final 出正式图。
      注意 draft 和 final 即使同 seed 构图也不同，draft 不能"精修"成 final，只能用来试提示词。

    aspect 按用途选：海报/手机壁纸 3:4 或 9:16，横幅/桌面 16:9，头像/图标 1:1。
    seed 不传则随机；想复现同一张图时传回上次返回的 seed。
    caller 填调用方标识（如 claude-code / hermes-xkc），只用于输出文件归档，不影响排队顺序。
    """
    w, h = SIZES[aspect][quality]
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
    graph = t2i_graph(prompt, w, h, STEPS[quality], seed, _prefix("gen", caller))
    im = await _run(graph, "generate", quality, ctx)
    data, host_path, url = await _fetch_result(im)
    preview, full = _preview(data)
    return [preview, f"已生成 {full[0]}x{full[1]}，{STEPS[quality]} 步，seed={seed}，耗时 {im['elapsed']:.0f}s。\n"
                     f"原图（Unraid）: {host_path}\nURL: {url}\n上面是缩小到 1024 的预览。"]


@mcp.tool()
async def edit_image(
    image: str,
    prompt: str,
    ctx: Context,
    quality: Literal["final", "draft"] = "final",
    seed: int | None = None,
    caller: str = "unknown",
) -> list:
    """用 Qwen-Image-2.1 编辑一张已有图片：换物体、改风格、改图中文字、扩展画面、做成海报等。输出画布跟随参考图的宽高比。

    image 接受三种形式：http(s) URL；base64（可带 data:image/...;base64, 前缀）；或 Unraid 上的文件路径（/mnt/user/... 开头，须是容器挂载可见的）。
    prompt 直接描述"要变成什么"，模型会保留参考图的构图和人物身份。一次一个明确改动最稳；要做多处修改就串联多次调用，把上一次的输出 URL 当下一次的 image。
    quality：final = 参考图放大到 2K 规模处理（约 5 分钟）；draft = 1024 规模（约 1.5 分钟），试提示词时用。
    """
    data = await _load_image_bytes(image)
    name = await _upload(data)
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
    graph = edit_graph(prompt, name, EDIT_REF_RES[quality], EDIT_STEPS[quality], seed, _prefix("edit", caller))
    im = await _run(graph, "edit", quality, ctx)
    out, host_path, url = await _fetch_result(im)
    preview, full = _preview(out)
    return [preview, f"已编辑，输出 {full[0]}x{full[1]}，{EDIT_STEPS[quality]} 步，seed={seed}，耗时 {im['elapsed']:.0f}s。\n"
                     f"原图（Unraid）: {host_path}\nURL: {url}\n上面是缩小到 1024 的预览。"]


@mcp.tool()
async def queue_status() -> str:
    """查看 ComfyUI 当前队列：正在执行几个、排队几个、大致要等多久。提交前不需要调用（generate/edit 会自己检查），只在用户问"现在忙不忙"时用。"""
    running, pending = await _queue_state()
    est = (running + pending) * EST_SECONDS[("generate", "final")]
    return f"执行中 {running} 个，排队 {pending} 个（上限 {MAX_PENDING}）。按每张 final 约 4 分钟估，新任务大约 {est // 60} 分钟后开始。"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT, streamable_http_path="/mcp", stateless_http=True)
