#!/usr/bin/env python3
"""comfyui-mcp — 把 ComfyUI 上的 Qwen-Image-2.1 暴露成 MCP（streamable-http）。

工具: generate_image / edit_image / queue_status
特点:
- 无状态；所有调用方共用 ComfyUI 的单 GPU FIFO 队列
- 等待期间通过 MCP progress 通知汇报排队位置和采样步数（客户端不会超时）
- 队列过长直接拒绝，防止 agent 循环提交
- 返回全分辨率图片（JPEG）+ PNG 原图的签名 URL；本机图片经一次性签名 URL 上传，不需要 ssh
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
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

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188").rstrip("/")          # 容器内访问 ComfyUI
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://10.10.10.2:8189").rstrip("/")      # 本服务对调用方的地址（图片 URL 前缀）
# 鉴权：MCP_AUTH_TOKENS="name1:token1,name2:token2"。为空则不鉴权（只应在纯内网用）
AUTH_TOKENS = {t.split(":", 1)[1].strip(): t.split(":", 1)[0].strip()
               for t in os.environ.get("MCP_AUTH_TOKENS", "").split(",") if ":" in t}
IMAGE_SIGN_KEY = os.environ.get("IMAGE_SIGN_KEY") or (next(iter(AUTH_TOKENS)) if AUTH_TOKENS else "dev")
MAX_PENDING = int(os.environ.get("MAX_PENDING", "5"))
PORT = int(os.environ.get("PORT", "8189"))

UNET = os.environ.get("UNET_NAME", "qwen_image_2.1_int8_convrot.safetensors")
CLIP = os.environ.get("CLIP_NAME", "qwen3vl_8b_int8_convrot.safetensors")
VAE = os.environ.get("VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors")

# 1K 为默认（4060 Ti 上 1 分钟内）；2k 是官方 README 的 2K 尺寸表，4 分钟一张，必须显式指定
_1K = {"1:1": (1024, 1024), "4:3": (1216, 896), "3:4": (896, 1216), "3:2": (1280, 864), "2:3": (864, 1280), "16:9": (1376, 768), "9:16": (768, 1376)}
_2K = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}
SIZES = {a: {"draft": _1K[a], "final": _1K[a], "2k": _2K[a]} for a in _1K}
STEPS = {"draft": 20, "final": 40, "2k": 40}
EDIT_STEPS = {"draft": 20, "final": 40}
EDIT_REF_RES = {"draft": 1024, "final": 1024}   # 编辑一律 1024 规模：1536 要 4 分钟、2048 要 10 分钟，用户不接受
# 实测（RTX 4060 Ti 16GB, int8）用于预估等待时间，单位秒
EST_SECONDS = {("generate", "draft"): 30, ("generate", "final"): 60, ("generate", "2k"): 240, ("edit", "draft"): 50, ("edit", "final"): 90}

INSTRUCTIONS = """本服务在家里 Unraid 的 RTX 4060 Ti 上跑 Qwen-Image-2.1（文生图 + 图片编辑，官方 int8 重打包权重），通过 ComfyUI 执行。

何时用：用户要生成图片、海报、插画、示意图，或要修改一张已有图片（换物体、改风格、改文字、扩展画面）。

必须知道的限制：
- 单 GPU、单队列：一次只能出一张图，所有调用方共用同一条 FIFO 队列。耗时（不含排队）：文生图 final(1K) 约 1 分钟、draft 约 30 秒、2k 约 4 分钟；编辑 final 约 1.5 分钟、draft 约 50 秒。工具会阻塞到出图为止并持续汇报进度，**不要中途取消、不要重复提交**。若调用被取消，正在跑的任务仍会跑完，图可在 ComfyUI 队列面板找到。
- 队列里已有 >5 个任务时工具会直接拒绝并给出预估等待，此时告诉用户稍后再试，不要循环重试。

提示词写法（实测结论）：
- 图里要出现的文字必须用引号原样写出（中英文都一样），如：招牌上写着"老陈肠粉"。不写具体文字，模型会画出乱码笔画。
- 中文场景用中文写提示词，模型的文本编码器对中文是原生的。
- 说清楚、不要堆砌：具体描述每个区域是什么媒介/材质/内容（"上半是真实照片，下半是米白亚麻布上的贴布绣"），比堆一串风格形容词有效得多。
- 负面提示词在当前配置（cfg 1.0）下不生效，不要依赖它。
- 编辑模式会锚定参考图的构图和人物身份；一次只描述一个明确的改动最稳，多个改动请分多次调用串联。
"""

class StaticTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        name = AUTH_TOKENS.get(token)
        return AccessToken(token=token, client_id=name, scopes=["image"]) if name else None


import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

if AUTH_TOKENS:
    mcp = MCPServer(
        "qwen-image", instructions=INSTRUCTIONS,
        token_verifier=StaticTokenVerifier(),
        auth=AuthSettings(issuer_url=AnyHttpUrl(PUBLIC_URL), resource_server_url=AnyHttpUrl(PUBLIC_URL + "/mcp"), required_scopes=["image"]),
    )
else:
    mcp = MCPServer("qwen-image", instructions=INSTRUCTIONS)


def _caller_name(fallback: str) -> str:
    tok = get_access_token()
    return tok.client_id if tok and tok.client_id else fallback


# ---------- 图片下载（签名 URL，不暴露 ComfyUI） ----------

def _sign(sub: str, filename: str) -> str:
    return hmac.new(IMAGE_SIGN_KEY.encode(), f"{sub}/{filename}".encode(), hashlib.sha256).hexdigest()[:32]


def _image_url(sub: str, filename: str) -> str:
    path = f"{sub}/{filename}" if sub else filename
    return f"{PUBLIC_URL}/image/{path}?sig={_sign(sub, filename)}"


@mcp.custom_route("/image/{path:path}", methods=["GET"])
async def image_route(request: Request) -> Response:
    """代理 ComfyUI 的 /view，只放行带正确 HMAC 签名的 URL；这样对外只需暴露本服务一个域名"""
    path = request.path_params["path"]
    sub, _, filename = path.rpartition("/")
    if ".." in path or not filename or not hmac.compare_digest(request.query_params.get("sig", ""), _sign(sub, filename)):
        return JSONResponse({"error": "invalid signature"}, status_code=403)
    async with _client() as c:
        r = await c.get("/view", params={"filename": filename, "subfolder": sub, "type": "output"})
    if r.status_code != 200:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(r.content, media_type=r.headers.get("content-type", "image/png"))


def _upload_sig(uid: str, exp: int) -> str:
    return hmac.new(IMAGE_SIGN_KEY.encode(), f"upload/{uid}/{exp}".encode(), hashlib.sha256).hexdigest()[:32]


@mcp.custom_route("/upload", methods=["POST"])
async def upload_route(request: Request) -> Response:
    """一次性签名上传：request_upload() 签发 URL，客户端 curl -F image=@file 上传，返回 ComfyUI 里的文件名"""
    uid, exp, sig = request.query_params.get("id", ""), request.query_params.get("exp", "0"), request.query_params.get("sig", "")
    if not uid.isalnum() or not exp.isdigit() or int(exp) < time.time() or not hmac.compare_digest(sig, _upload_sig(uid, int(exp))):
        return JSONResponse({"error": "invalid or expired upload url"}, status_code=403)
    form = await request.form()
    f = form.get("image")
    if f is None:
        return JSONResponse({"error": "multipart field 'image' required"}, status_code=400)
    data = await f.read()
    if len(data) > 30_000_000:
        return JSONResponse({"error": "image too large (max 30 MB)"}, status_code=413)
    try:
        PILImage.open(io.BytesIO(data)).verify()
    except Exception:
        return JSONResponse({"error": "not an image"}, status_code=400)
    name = await _upload(data)
    return JSONResponse({"image": name, "hint": f"传给 edit_image 的 image 参数: {name}"})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    try:
        running, pending = await _queue_state()
        return JSONResponse({"status": "ok", "auth": bool(AUTH_TOKENS), "queue_running": running, "queue_pending": pending})
    except Exception as e:  # ComfyUI 不可达
        return JSONResponse({"status": "comfyui_unreachable", "error": str(e)[:200]}, status_code=503)


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


UPLOADED_NAME = re.compile(r"^mcp_[0-9a-f]{12}\.png$")


async def _load_image_bytes(image: str) -> bytes:
    """支持：http(s) URL / data URI 或裸 base64 / 容器可见路径。已上传的 mcp_xxx.png 文件名在调用处直接用，不经这里"""
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
                        el = time.time() - t0
                        eta = (el / step) * (total_steps - step) if step else 0
                        await ctx.report_progress(step, total_steps, f"采样 {step}/{total_steps} 步 · 已用 {el:.0f}s · 预计还需 {eta:.0f}s")
                        continue
                    if mt == "execution_error" and d.get("prompt_id") == pid:
                        raise ToolError(f"ComfyUI 执行出错: {d.get('exception_message', '')[:600]}")
                    if mt == "execution_success" and d.get("prompt_id") == pid:
                        break
                pos = await _queue_position(pid)
                if pos is None:
                    break
                if pos > 0:
                    await ctx.report_progress(0, total_steps, f"排队中，前面还有 {pos} 个任务（每个约 {EST_SECONDS[(kind, quality)] // 60} 分钟）· 已等 {time.time() - t0:.0f}s")
    except asyncio.CancelledError:
        # 调用方放弃等待：只撤销还在排队的任务；正在跑的让它跑完落盘（几分钟的活杀掉太亏，图可从 ComfyUI 队列面板拿）
        async with _client() as c:
            pos = await _queue_position(pid)
            if pos and pos > 0:
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


async def _fetch_result(im: dict) -> tuple[bytes, str]:
    """返回 (原图字节, 签名 URL)"""
    sub = im.get("subfolder", "")
    q = f"filename={im['filename']}&subfolder={sub}&type={im['type']}"
    async with _client() as c:
        data = (await c.get(f"/view?{q}")).content
    return data, _image_url(sub, im["filename"])


def _full_jpeg(data: bytes) -> tuple[Image, tuple[int, int]]:
    """全分辨率返回。Claude API 单图上限 5 MB，2K PNG 约 7.5 MB 传不过去，所以图片内容用全分辨率 JPEG，PNG 原文件走 URL"""
    img = PILImage.open(io.BytesIO(data))
    full = img.size
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92, optimize=True)
    if buf.tell() > 4_500_000:  # 极端情况下再降质量保证不超限
        buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=80, optimize=True)
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
    quality: Literal["final", "draft", "2k"] = "final",
    seed: int | None = None,
    caller: str = "unknown",
) -> list:
    """用 Qwen-Image-2.1 文生图。阻塞直到出图（final 约 1 分钟，draft 约 30 秒，2k 约 4 分钟，另加排队时间），期间持续汇报进度；不要中途取消。

    quality 怎么选：
    - final（默认）：1K（1024² 级）、40 步。用户要一张能用的图 → 用这个。
    - draft：1K、20 步。用户说"先看看 / 快速试试 / 出几个方向"，或同一提示词要连续改多版 → 用这个。
    - 2k：官方 2K 尺寸表（2048² 级）、40 步，约 4 分钟。**只在用户明确要求 2K / 大图 / 打印用途时才用**，否则不要选。
      注意各档即使同 seed 构图也不同，draft 不能"精修"成 final。

    aspect 按用途选：海报/手机壁纸 3:4 或 9:16，横幅/桌面 16:9，头像/图标 1:1。
    seed 不传则随机；想复现同一张图时传回上次返回的 seed。
    caller 填调用方标识（如 claude-code / hermes-xkc），只用于输出文件归档；开启鉴权时自动取 token 对应的名字，可不填。
    """
    caller = _caller_name(caller)
    w, h = SIZES[aspect][quality]
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
    graph = t2i_graph(prompt, w, h, STEPS[quality], seed, _prefix("gen", caller))
    im = await _run(graph, "generate", quality, ctx)
    data, url = await _fetch_result(im)
    img, full = _full_jpeg(data)
    return [img, f"已生成 {full[0]}x{full[1]}，{STEPS[quality]} 步，seed={seed}，耗时 {im['elapsed']:.0f}s。\nPNG 原图: {url}"]


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

    image 接受：① 本机文件 → 先调 request_upload() 拿到上传命令，执行后得到的文件名（mcp_xxx.png）；② http(s) URL（包括本服务之前返回的 PNG 原图 URL）；③ base64。
    绝不要用 ssh/scp 把文件搬到服务器，也不要传服务器路径。
    prompt 直接描述"要变成什么"，模型会保留参考图的构图和人物身份。一次一个明确改动最稳；要做多处修改就串联多次调用，把上一次的输出 URL 当下一次的 image。
    quality：final（默认）= 1024 规模、40 步，约 1.5 分钟；draft = 20 步，约 50 秒，试提示词用（细节略糙）。编辑不提供 2K。
    """
    caller = _caller_name(caller)
    if UPLOADED_NAME.match(image):
        name = image                      # request_upload → curl 上传后返回的文件名
    else:
        name = await _upload(await _load_image_bytes(image))
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
    graph = edit_graph(prompt, name, EDIT_REF_RES[quality], EDIT_STEPS[quality], seed, _prefix("edit", caller))
    im = await _run(graph, "edit", quality, ctx)
    out, url = await _fetch_result(im)
    img, full = _full_jpeg(out)
    return [img, f"已编辑，输出 {full[0]}x{full[1]}，{EDIT_STEPS[quality]} 步，seed={seed}，耗时 {im['elapsed']:.0f}s。\nPNG 原图: {url}"]


@mcp.tool()
async def request_upload() -> str:
    """要编辑一张本机（调用方电脑上）的图片时先调这个：返回一条 curl 命令，用它把文件上传（URL 带 10 分钟有效的一次性签名，不含任何密钥），
    命令输出的 JSON 里 "image" 字段就是 edit_image 的 image 参数。不要用 ssh/scp。"""
    uid, exp = uuid.uuid4().hex[:16], int(time.time()) + 600
    url = f"{PUBLIC_URL}/upload?id={uid}&exp={exp}&sig={_upload_sig(uid, exp)}"
    return (f"在调用方本机执行（把 <file> 换成图片路径）：\n"
            f"curl -sS -F image=@<file> '{url}'\n"
            f"返回 JSON 的 image 字段（形如 mcp_xxxxxxxxxxxx.png）传给 edit_image。URL 10 分钟内有效，可重复使用。")


@mcp.tool()
async def queue_status() -> str:
    """查看 ComfyUI 当前队列：正在执行几个、排队几个、大致要等多久。提交前不需要调用（generate/edit 会自己检查），只在用户问"现在忙不忙"时用。"""
    running, pending = await _queue_state()
    est = (running + pending) * EST_SECONDS[("generate", "final")]
    return f"执行中 {running} 个，排队 {pending} 个（上限 {MAX_PENDING}）。按每张 final 约 4 分钟估，新任务大约 {est // 60} 分钟后开始。"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT, streamable_http_path="/mcp", stateless_http=True)
