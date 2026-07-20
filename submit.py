"""
Usage:
    python submit.py image.png                           # single image -> 3D
    python submit.py image.png --stl                     # output STL instead of GLB
    python submit.py front.png side.png back.png         # multi-angle -> single 3D mesh
    python submit.py *.png --mode multidiffusion         # multi-angle with multidiffusion blending
    python submit.py --batch                             # submit all images in input/ as separate jobs
    python submit.py --batch --workers 5                 # limit to 5 concurrent batch jobs
    python submit.py --stress 10                         # hammer endpoint with nelson.png x10 concurrently
    python submit.py nelson.png --sweep                  # run all preset configs concurrently and compare
    python submit.py nelson.png --sparse-steps 16 --slat-cfg 3.5 --simplify 0.97  # custom params

Sweep presets (--sweep):
    original   slat_cfg=3.5  slat_steps=12  sparse_steps=12  simplify=0.97  (TRELLIS paper defaults)
    low_cfg    slat_cfg=3.5  slat_steps=25  sparse_steps=25  simplify=0.97
    mid_cfg    slat_cfg=5.0  slat_steps=25  sparse_steps=25  simplify=0.97
    high_steps slat_cfg=3.5  slat_steps=50  sparse_steps=50  simplify=0.97
    current    slat_cfg=7.5  slat_steps=50  sparse_steps=50  simplify=1.0   (current broken defaults)

Requires:
    pip install requests python-dotenv
    RUNPOD_API_KEY and ENDPOINT_ID in .env
"""

import argparse
import base64
import gzip
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

SWEEP_PRESETS = {
    "original":   {"sparse_steps": 12, "sparse_cfg": 7.5, "slat_steps": 12, "slat_cfg": 3.5, "simplify_ratio": 0.97},
    "low_cfg":    {"sparse_steps": 25, "sparse_cfg": 7.5, "slat_steps": 25, "slat_cfg": 3.5, "simplify_ratio": 0.97},
    "mid_cfg":    {"sparse_steps": 25, "sparse_cfg": 7.5, "slat_steps": 25, "slat_cfg": 5.0, "simplify_ratio": 0.97},
    "high_steps": {"sparse_steps": 50, "sparse_cfg": 7.5, "slat_steps": 50, "slat_cfg": 3.5, "simplify_ratio": 0.97},
    "current":    {"sparse_steps": 50, "sparse_cfg": 7.5, "slat_steps": 50, "slat_cfg": 7.5, "simplify_ratio": 1.0},
}


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def build_input(image_path: str, output_format: str, params: dict) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"image": b64, "output_format": output_format, **params}


def build_multi_input(image_paths: list, output_format: str, mode: str, params: dict) -> dict:
    b64_list = []
    for path in image_paths:
        with open(path, "rb") as f:
            b64_list.append(base64.b64encode(f.read()).decode())
    return {"images": b64_list, "output_format": output_format, "mode": mode, **params}


def submit(payload: dict) -> str:
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": payload},
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{stem}__{ts}.{fmt}")
    mesh_bytes = base64.b64decode(output["mesh_b64"])
    if output.get("compressed"):
        mesh_bytes = gzip.decompress(mesh_bytes)
    with open(out_path, "wb") as f:
        f.write(mesh_bytes)
    return out_path, output.get("face_count", "?"), output.get("vertex_count", "?")


def run_job(image_path: str, fmt: str, params: dict, label: str = "") -> dict:
    label = label or os.path.basename(image_path)
    t0 = time.time()
    try:
        payload = build_input(image_path, fmt, params)
        job_id = submit(payload)
        log(f"  [{label}] submitted -> {job_id}")
        result = poll(job_id, label=label)
        elapsed = time.time() - t0

        if result["status"] != "COMPLETED":
            return {"label": label, "ok": False, "error": result.get("error", "unknown"), "elapsed": elapsed}

        stem = os.path.splitext(os.path.basename(image_path))[0]
        if label != os.path.basename(image_path):
            stem = f"{stem}__{label}"
        out_path, faces, verts = save_mesh(result["output"], stem, fmt)
        return {"label": label, "ok": True, "path": out_path, "faces": faces, "verts": verts, "elapsed": elapsed, "params": params}

    except Exception as e:
        return {"label": label, "ok": False, "error": str(e), "elapsed": time.time() - t0}


def run_multi_job(image_paths: list, fmt: str, mode: str, params: dict, label: str = "") -> dict:
    label = label or "+".join(os.path.basename(p) for p in image_paths)
    t0 = time.time()
    try:
        payload = build_multi_input(image_paths, fmt, mode, params)
        job_id = submit(payload)
        log(f"  [{label}] submitted ({len(image_paths)} images, mode={mode}) -> {job_id}")
        result = poll(job_id, label=label)
        elapsed = time.time() - t0

        if result["status"] != "COMPLETED":
            return {"label": label, "ok": False, "error": result.get("error", "unknown"), "elapsed": elapsed}

        stem = "multi_" + "_".join(os.path.splitext(os.path.basename(p))[0] for p in image_paths)
        out_path, faces, verts = save_mesh(result["output"], stem, fmt)
        return {"label": label, "ok": True, "path": out_path, "faces": faces, "verts": verts, "elapsed": elapsed, "params": params}

    except Exception as e:
        return {"label": label, "ok": False, "error": str(e), "elapsed": time.time() - t0}


def print_summary(results: list):
    ok   = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"\n{'='*60}")
    print(f"Results: {len(ok)} succeeded, {len(fail)} failed")
    for r in sorted(ok, key=lambda x: x["label"]):
        p = r.get("params", {})
        param_str = ""
        if p:
            param_str = (f"  slat_cfg={p.get('slat_cfg','?')}  slat_steps={p.get('slat_steps','?')}"
                         f"  sparse_steps={p.get('sparse_steps','?')}  simplify={p.get('simplify_ratio','?')}")
        print(f"  OK   [{r['label']}] {r['faces']} faces  {r['elapsed']:.1f}s{param_str}")
        print(f"       -> {r['path']}")
    for r in fail:
        print(f"  FAIL [{r['label']}] {r['error']}  {r['elapsed']:.1f}s")
    if ok:
        avg = sum(r["elapsed"] for r in ok) / len(ok)
        print(f"  Avg time: {avg:.1f}s")
    print('='*60)


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
    parser.add_argument("--workers", type=int, default=0, help="Max concurrent jobs for batch/sweep (0 = all)")
    parser.add_argument("--stress", type=int, default=0, metavar="N",
                        help="Stress test: submit the same image N times concurrently")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all preset configs concurrently on the given image and compare")

    # Sampler params
    parser.add_argument("--sparse-steps", type=int,   default=None, help="Sparse structure sampler steps")
    parser.add_argument("--sparse-cfg",   type=float, default=None, help="Sparse structure CFG strength")
    parser.add_argument("--slat-steps",   type=int,   default=None, help="SLat sampler steps")
    parser.add_argument("--slat-cfg",     type=float, default=None, help="SLat CFG strength (try 3.5 for clean results)")
    parser.add_argument("--simplify",     type=float, default=None, help="Mesh simplification ratio (0-1, e.g. 0.97)")
    parser.add_argument("--texture-size", type=int,   default=None, help="Texture resolution (default 2048)")
    parser.add_argument("--seed",         type=int,   default=None, help="Random seed")

    args = parser.parse_args()
    fmt = "stl" if args.stl else "glb"

    # Build params dict from CLI flags; only include keys that were explicitly set
    cli_params = {}
    if args.sparse_steps is not None: cli_params["sparse_steps"]   = args.sparse_steps
    if args.sparse_cfg   is not None: cli_params["sparse_cfg"]     = args.sparse_cfg
    if args.slat_steps   is not None: cli_params["slat_steps"]     = args.slat_steps
    if args.slat_cfg     is not None: cli_params["slat_cfg"]       = args.slat_cfg
    if args.simplify     is not None: cli_params["simplify_ratio"] = args.simplify
    if args.texture_size is not None: cli_params["texture_size"]   = args.texture_size
    if args.seed         is not None: cli_params["seed"]           = args.seed

    # --- sweep mode ---
    if args.sweep:
        image_path = resolve_image(args.images[0] if args.images else "nelson.png")
        presets = SWEEP_PRESETS
        max_workers = args.workers or len(presets)
        print(f"Sweep: {len(presets)} configs on {image_path}  ({max_workers} concurrent)")
        for name, p in presets.items():
            print(f"  {name:12s}  slat_cfg={p['slat_cfg']}  slat_steps={p['slat_steps']}"
                  f"  sparse_steps={p['sparse_steps']}  simplify={p['simplify_ratio']}")
        print()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_job, image_path, fmt, p, name): name for name, p in presets.items()}
            results = [f.result() for f in as_completed(futures)]
        print_summary(results)
        print(f"Total wall time: {time.time() - t0:.1f}s")
        return

    # --- stress test mode ---
    if args.stress:
        image_path = resolve_image(args.images[0] if args.images else "nelson.png")
        n = args.stress
        print(f"Stress test: submitting {n} concurrent jobs for {image_path}")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(run_job, image_path, fmt, cli_params, str(i + 1)) for i in range(n)]
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
            futures = {ex.submit(run_job, img, fmt, cli_params, os.path.basename(img)): img for img in images}
            results = [f.result() for f in as_completed(futures)]
        print_summary(results)
        print(f"Total wall time: {time.time() - t0:.1f}s")
        return

    if not args.images:
        parser.error("Provide one image (single) or 2+ images (multi-angle), or use --batch / --sweep / --stress N")

    image_paths = [resolve_image(p) for p in args.images]

    # --- multi-angle mode ---
    if len(image_paths) >= 2:
        print(f"Multi-angle: {len(image_paths)} images, mode={args.mode}, format={fmt}")
        result = run_multi_job(image_paths, fmt, mode=args.mode, params=cli_params)
        if result["ok"]:
            print(f"Done -> {result['path']}  ({result['faces']} faces, {result['verts']} verts)  {result['elapsed']:.1f}s")
        else:
            sys.exit(f"Job failed: {result['error']}")
        return

    # --- single image mode ---
    if cli_params:
        print(f"Params: {cli_params}")
    print(f"Submitting {image_paths[0]} -> {fmt}")
    result = run_job(image_paths[0], fmt, cli_params)
    if result["ok"]:
        print(f"Done -> {result['path']}  ({result['faces']} faces, {result['verts']} verts)  {result['elapsed']:.1f}s")
    else:
        sys.exit(f"Job failed: {result['error']}")


if __name__ == "__main__":
    main()
