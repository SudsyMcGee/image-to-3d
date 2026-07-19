"""
Test Zero123++ endpoint with a single image.

Usage:
    python test_zero.py input/yourfile.webp
    python test_zero.py input/yourfile.webp --steps 75 --guidance 4.0
    python test_zero.py input/yourfile.webp --no-rembg

Output saved to output/zero_grid.png and output/zero_view_0..5.png

Requires RUNPOD_API_KEY and MULTIVIEW_ENDPOINT_ID in .env
"""

import argparse
import base64
import io
import os
import sys
import time

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("MULTIVIEW_ENDPOINT_ID", "")

if not API_KEY:
    sys.exit("RUNPOD_API_KEY not set in .env")
if not ENDPOINT_ID:
    sys.exit("MULTIVIEW_ENDPOINT_ID not set in .env")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def submit(image_path: str, steps: int, guidance: float, prompt: str, remove_bg: bool) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": {
            "image":     b64,
            "steps":     steps,
            "guidance":  guidance,
            "prompt":    prompt,
            "remove_bg": remove_bg,
        }},
        timeout=30,
    )
    if not resp.ok:
        sys.exit(f"Submit failed: {resp.status_code} {resp.text}")
    return resp.json()["id"]


def poll(job_id: str) -> dict:
    url = f"{BASE_URL}/status/{job_id}"
    last = None
    while True:
        data = requests.get(url, headers=HEADERS, timeout=30).json()
        status = data.get("status")
        if status != last:
            print(f"  [{job_id[:8]}] {status}", flush=True)
            last = status
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return data
        time.sleep(4)


def save_outputs(output: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    grid = Image.open(io.BytesIO(base64.b64decode(output["grid_b64"])))
    grid_path = os.path.join(out_dir, "zero_grid.png")
    grid.save(grid_path)
    print(f"  Grid saved -> {grid_path}")

    for i, v_b64 in enumerate(output["views_b64"]):
        view = Image.open(io.BytesIO(base64.b64decode(v_b64)))
        vpath = os.path.join(out_dir, f"zero_view_{i}.png")
        view.save(vpath)
        print(f"  View {i}  -> {vpath}")


def main():
    parser = argparse.ArgumentParser(description="Test Zero123++ endpoint")
    parser.add_argument("image", help="Path to input image (PNG/JPG/WEBP)")
    parser.add_argument("--steps",    type=int,   default=75,  help="Inference steps (default 75)")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale (default 4.0)")
    parser.add_argument("--prompt",   type=str,   default="",  help="Optional text nudge")
    parser.add_argument("--no-rembg", action="store_true",     help="Skip background removal")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"File not found: {args.image}")

    print(f"Submitting {args.image}  steps={args.steps}  guidance={args.guidance}  rembg={not args.no_rembg}")
    t0 = time.time()

    job_id = submit(args.image, args.steps, args.guidance, args.prompt, not args.no_rembg)
    print(f"Job submitted -> {job_id}")

    result = poll(job_id)
    elapsed = time.time() - t0

    if result["status"] != "COMPLETED":
        sys.exit(f"Job failed: {result.get('error', result)}")

    out = result["output"]
    if "error" in out:
        sys.exit(f"Handler error: {out['error']}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_outputs(out, out_dir)
    print(f"\nDone in {elapsed:.1f}s  (steps={out['steps']} guidance={out['guidance']})")


if __name__ == "__main__":
    main()
