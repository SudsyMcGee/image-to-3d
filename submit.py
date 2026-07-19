"""
Usage:
    python submit.py image.png                           # single image -> 3D
    python submit.py image.png --stl                     # output STL instead of GLB
    python submit.py front.png side.png back.png         # multi-angle -> single 3D mesh
    python submit.py *.png --mode multidiffusion         # multi-angle with multidiffusion blending
    python submit.py --batch                             # submit all images in input/ as separate jobs
    python submit.py --batch --workers 5                 # limit to 5 concurrent batch jobs
    python submit.py --stress 10                         # hammer endpoint with nelson.png x10 concurrently

Requires:
    pip install requests python-dotenv
    RUNPOD_API_KEY and ENDPOINT_ID in .env
"""

import argparse
import base64
import glob
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "")

if not API_KEY:
    sys.exit("RUNPOD_API_KEY not set in .env")
if not ENDPOINT_ID:
    sys.exit("ENDPOINT_ID not set in .env")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

_print_lock = threading.Lock()

def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def submit_job(image_path: str, output_format: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": {"image": b64, "output_format": output_format}},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()["id"]


def submit_multi_job(image_paths: list, output_format: str, mode: str = "stochastic") -> str:
    b64_list = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64_list.append(base64.b64encode(f.read()).decode())
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": {"images": b64_list, "output_format": output_format, "mode": mode}},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()["id"]


def poll(job_id: str, label: str = "") -> dict:
    url = f"{BASE_URL}/status/{job_id}"
    last_status = None
    while True:
        data = requests.get(url, headers=HEADERS, timeout=30).json()
        status = data.get("status")
        if status != last_status:
            log(f"  [{label or job_id[:8]}] {status}")
            last_status = status
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return data
        time.sleep(4)


def save_mesh(output: dict, stem: str, fmt: str) -> tuple:
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.{fmt}")
    mesh_bytes = base64.b64decode(output["mesh_b64"])
    with open(out_path, "wb") as f:
        f.write(mesh_bytes)
    return out_path, output.get("face_count", "?"), output.get("vertex_count", "?")


def run_job(image_path: str, fmt: str, label: str = "") -> dict:
    """Submit one job and wait for it. Returns a result summary dict."""
    label = label or os.path.basename(image_path)
    t0 = time.time()
    try:
        job_id = submit_job(image_path, fmt)
        log(f"  [{label}] submitted -> {job_id}")
        result = poll(job_id, label=label)
        elapsed = time.time() - t0

        if result["status"] != "COMPLETED":
            return {"label": label, "ok": False, "error": result.get("error", "unknown"), "elapsed": elapsed}

        stem = os.path.splitext(os.path.basename(image_path))[0]
        # avoid collisions in batch mode by appending label suffix if needed
        if label != stem:
            stem = f"{stem}_{label}"
        out_path, faces, verts = save_mesh(result["output"], stem, fmt)
        return {"label": label, "ok": True, "path": out_path, "faces": faces, "verts": verts, "elapsed": elapsed}

    except Exception as e:
        return {"label": label, "ok": False, "error": str(e), "elapsed": time.time() - t0}


def run_multi_job(image_paths: list, fmt: str, mode: str = "stochastic", label: str = "") -> dict:
    """Submit a multi-image job and wait for it."""
    label = label or "+".join(os.path.basename(p) for p in image_paths)
    t0 = time.time()
    try:
        job_id = submit_multi_job(image_paths, fmt, mode)
        log(f"  [{label}] submitted ({len(image_paths)} images, mode={mode}) -> {job_id}")
        result = poll(job_id, label=label)
        elapsed = time.time() - t0

        if result["status"] != "COMPLETED":
            return {"label": label, "ok": False, "error": result.get("error", "unknown"), "elapsed": elapsed}

        stem = "multi_" + "_".join(os.path.splitext(os.path.basename(p))[0] for p in image_paths)
        out_path, faces, verts = save_mesh(result["output"], stem, fmt)
        return {"label": label, "ok": True, "path": out_path, "faces": faces, "verts": verts, "elapsed": elapsed}

    except Exception as e:
        return {"label": label, "ok": False, "error": str(e), "elapsed": time.time() - t0}


def print_summary(results: list):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"\n{'='*50}")
    print(f"Results: {len(ok)} succeeded, {len(fail)} failed")
    for r in ok:
        print(f"  OK  [{r['label']}] {r['path']}  ({r['faces']} faces, {r['verts']} verts)  {r['elapsed']:.1f}s")
    for r in fail:
        print(f"  FAIL [{r['label']}] {r['error']}  {r['elapsed']:.1f}s")
    if ok:
        avg = sum(r["elapsed"] for r in ok) / len(ok)
        print(f"  Avg time (successful): {avg:.1f}s")
    print('='*50)


def resolve_image(path: str) -> str:
    if os.path.exists(path):
        return path
    candidate = os.path.join(os.path.dirname(__file__), "input", path)
    if os.path.exists(candidate):
        return candidate
    sys.exit(f"File not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Submit image(s) -> 3D mesh via RunPod")
    parser.add_argument("images", nargs="*", metavar="IMG", help="One image (single) or 2+ images (multi-angle)")
    parser.add_argument("--stl", action="store_true", help="Output STL instead of GLB")
    parser.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"],
                        help="Multi-angle blending mode (default: stochastic)")
    parser.add_argument("--batch", action="store_true", help="Submit all images in input/ as separate jobs")
    parser.add_argument("--workers", type=int, default=0, help="Max concurrent jobs for batch (0 = all)")
    parser.add_argument("--stress", type=int, default=0, metavar="N",
                        help="Stress test: submit the same image N times concurrently")
    args = parser.parse_args()

    fmt = "stl" if args.stl else "glb"

    # --- stress test mode ---
    if args.stress:
        image_path = resolve_image(args.images[0] if args.images else "nelson.png")
        n = args.stress
        print(f"Stress test: submitting {n} concurrent jobs for {image_path}")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(run_job, image_path, fmt, str(i + 1)) for i in range(n)]
            results = [f.result() for f in as_completed(futures)]
        print_summary(results)
        print(f"Total wall time: {time.time() - t0:.1f}s")
        return

    # --- batch mode ---
    if args.batch:
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        images = sorted(
            glob.glob(os.path.join(input_dir, "*.png")) +
            glob.glob(os.path.join(input_dir, "*.jpg")) +
            glob.glob(os.path.join(input_dir, "*.jpeg"))
        )
        if not images:
            sys.exit(f"No images found in {input_dir}/")
        max_workers = args.workers or len(images)
        print(f"Batch: {len(images)} images, {max_workers} concurrent workers")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_job, img, fmt, os.path.basename(img)): img for img in images}
            results = [f.result() for f in as_completed(futures)]
        print_summary(results)
        print(f"Total wall time: {time.time() - t0:.1f}s")
        return

    if not args.images:
        parser.error("Provide one image (single) or 2+ images (multi-angle), or use --batch / --stress N")

    image_paths = [resolve_image(p) for p in args.images]

    # --- multi-angle mode ---
    if len(image_paths) >= 2:
        print(f"Multi-angle: {len(image_paths)} images, mode={args.mode}, format={fmt}")
        result = run_multi_job(image_paths, fmt, mode=args.mode)
        if result["ok"]:
            print(f"Done -> {result['path']}  ({result['faces']} faces, {result['verts']} verts)  {result['elapsed']:.1f}s")
        else:
            sys.exit(f"Job failed: {result['error']}")
        return

    # --- single image mode ---
    print(f"Submitting {image_paths[0]} -> {fmt}")
    result = run_job(image_paths[0], fmt)
    if result["ok"]:
        print(f"Done -> {result['path']}  ({result['faces']} faces, {result['verts']} verts)  {result['elapsed']:.1f}s")
    else:
        sys.exit(f"Job failed: {result['error']}")


if __name__ == "__main__":
    main()
