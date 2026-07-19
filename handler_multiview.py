"""
RunPod serverless handler: image -> Zero123++ -> 6 multi-view images

Input:
    {
        "image":      "<base64 PNG/JPG/WEBP>",  # required (or image_url)
        "image_url":  "<https://...>",
        "steps":      75,                        # inference steps (default: 75 = max quality)
        "guidance":   4.0,                       # guidance scale (default: 4.0)
        "prompt":     "",                        # optional text nudge (weak conditioning)
        "remove_bg":  true                       # run rembg before Zero123++ (default: true)
    }

Output:
    {
        "grid_b64":   "<base64 PNG — 640x960 full 2x3 grid>",
        "views_b64":  ["<base64>", ...],         # 6 individual 320x320 views
        "steps":      75,
        "guidance":   4.0
    }
"""

import os
import sys
import base64
import io
import traceback

import runpod
import numpy as np
from PIL import Image

_STARTUP_ERROR = None
pipeline = None

HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
os.environ["HF_HOME"] = HF_HOME

try:
    import torch
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    print("Loading Zero123++ pipeline...", flush=True)
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-v1.2",
        torch_dtype=torch.float16,
        cache_dir=HF_HOME,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    pipeline.to("cuda")
    print("Zero123++ ready.", flush=True)

except Exception:
    _STARTUP_ERROR = traceback.format_exc()
    print(f"STARTUP FAILED:\n{_STARTUP_ERROR}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(job_input: dict) -> Image.Image:
    if "image" in job_input:
        return Image.open(io.BytesIO(base64.b64decode(job_input["image"])))
    if "image_url" in job_input:
        import urllib.request
        with urllib.request.urlopen(job_input["image_url"]) as r:
            return Image.open(io.BytesIO(r.read()))
    raise ValueError("Input must contain 'image' (base64) or 'image_url'")


def remove_background(img: Image.Image) -> Image.Image:
    import rembg
    session = rembg.new_session("u2net")
    return rembg.remove(img.convert("RGB"), session=session)


def preprocess(img: Image.Image, do_rembg: bool) -> Image.Image:
    if do_rembg:
        img = remove_background(img)  # returns RGBA

    # Convert RGBA → RGB on white background so Zero123++ gets clean input
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")

    # Pad to square then resize to 320x320
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    return padded.resize((320, 320), Image.Resampling.LANCZOS)


def split_grid(grid: Image.Image) -> list:
    """Split 640x960 Zero123++ output grid into 6 individual 320x320 PIL images."""
    views = []
    for row in range(3):
        for col in range(2):
            x, y = col * 320, row * 320
            views.append(grid.crop((x, y, x + 320, y + 320)))
    return views


def img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    if _STARTUP_ERROR:
        return {"error": f"Worker startup failed: {_STARTUP_ERROR}"}

    inp = job.get("input", {})

    steps    = int(inp.get("steps", 75))
    guidance = float(inp.get("guidance", 4.0))
    prompt   = str(inp.get("prompt", ""))
    do_rembg = bool(inp.get("remove_bg", True))

    try:
        raw = load_image(inp)
    except Exception as e:
        return {"error": f"Image load failed: {e}"}

    try:
        img = preprocess(raw, do_rembg)
    except Exception as e:
        return {"error": f"Preprocessing failed: {e}"}

    try:
        result = pipeline(
            img,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        grid: Image.Image = result.images[0]
        views = split_grid(grid)

        return {
            "grid_b64":  img_to_b64(grid),
            "views_b64": [img_to_b64(v) for v in views],
            "steps":     steps,
            "guidance":  guidance,
        }

    except Exception:
        return {"error": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
