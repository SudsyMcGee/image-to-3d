"""
Usage:
    python submit.py image.png            # outputs image.glb
    python submit.py image.png --stl      # outputs image.stl
    python submit.py image.png --watch    # poll and print status live

Requires:
    pip install requests python-dotenv
    RUNPOD_API_KEY and ENDPOINT_ID in .env
"""

import argparse
import base64
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "")

if not API_KEY:
    sys.exit("RUNPOD_API_KEY not set in .env")
if not ENDPOINT_ID:
    sys.exit("ENDPOINT_ID not set in .env — run deploy.ps1 deploy first, then add it")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def submit_job(image_path: str, output_format: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": {"image": b64, "output_format": output_format}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def poll(job_id: str, watch: bool) -> dict:
    url = f"{BASE_URL}/status/{job_id}"
    while True:
        data = requests.get(url, headers=HEADERS, timeout=30).json()
        status = data.get("status")
        if watch:
            print(f"\r  {status}...", end="", flush=True)
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            if watch:
                print()
            return data
        time.sleep(4)


def save_mesh(output: dict, image_path: str, fmt: str):
    stem = os.path.splitext(image_path)[0]
    out_path = f"{stem}.{fmt}"
    mesh_bytes = base64.b64decode(output["mesh_b64"])
    with open(out_path, "wb") as f:
        f.write(mesh_bytes)
    return out_path, output.get("face_count", "?"), output.get("vertex_count", "?")


def main():
    parser = argparse.ArgumentParser(description="Submit image → 3D mesh via RunPod")
    parser.add_argument("image", help="Path to input image (PNG/JPG)")
    parser.add_argument("--stl", action="store_true", help="Output STL instead of GLB")
    parser.add_argument("--watch", action="store_true", default=True,
                        help="Poll and show live status (default: on)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"File not found: {args.image}")

    fmt = "stl" if args.stl else "glb"

    print(f"Submitting {args.image} → {fmt} ...")
    job_id = submit_job(args.image, fmt)
    print(f"Job ID: {job_id}")

    result = poll(job_id, watch=args.watch)

    if result["status"] != "COMPLETED":
        sys.exit(f"Job failed: {result.get('error') or result}")

    out_path, faces, verts = save_mesh(result["output"], args.image, fmt)
    print(f"Done → {out_path}  ({faces} faces, {verts} vertices)")


if __name__ == "__main__":
    main()
