"""
RunPod serverless handler: image(s) (base64 or URL) -> TRELLIS -> GLB/STL (base64)

Single-image input:
    {
        "image": "<base64-encoded PNG/JPG>",   # required (or use image_url)
        "image_url": "<https://...>",           # alternative to image
        ...
    }

Multi-image input (multiple angles of the same object):
    {
        "images": ["<base64>", "<base64>", ...],     # list of base64 images
        "image_urls": ["<url>", "<url>", ...],        # alternative: list of URLs
        "mode": "stochastic" | "multidiffusion",      # default: "stochastic"
        ...
    }

Shared parameters:
        "output_format": "glb" | "stl",        # default: "glb"
        "simplify_ratio": 0.97,                 # mesh decimation (0-1, default 0.97)
        "texture_size": 2048,                   # texture resolution (default 2048)
        "seed": 42,                             # random seed (default 42)
        "sparse_steps": 16,                     # sparse structure sampler steps (default 16)
        "sparse_cfg": 7.5,                      # sparse structure cfg strength (default 7.5)
        "slat_steps": 16,                       # SLat sampler steps (default 16)
        "slat_cfg": 3.5                         # SLat cfg strength (default 3.5)

Output:
    {
        "mesh_b64": "<base64-encoded mesh file>",
        "format": "glb" | "stl",
        "vertex_count": <int>,
        "face_count": <int>
    }
"""

import os
import sys
import subprocess
import base64
import io
import tempfile
import traceback

import runpod
import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, "/app/TRELLIS")

MODEL_ID = os.environ.get("MODEL_ID", "JeffreyXiang/TRELLIS-image-large")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
os.environ["HF_HOME"] = HF_HOME  # ensure huggingface_hub uses our cache dir

_STARTUP_ERROR = None
pipeline = None

# Flag file stores image version — if it doesn't match or imports fail, recompile.
_EXT_FLAG = "/workspace/cuda_extensions_installed"
_IMAGE_VERSION = "v9-r2"


def _cuda_extensions_importable() -> bool:
    try:
        import nvdiffrast.torch  # noqa: F401
        import diffoctreerast    # noqa: F401
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except ImportError:
        return False


def _install_cuda_extensions():
    # nvcc lives in /usr/local/cuda/bin — must be on PATH for compilation to work
    cuda_bin = "/usr/local/cuda/bin"
    os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9"
    os.environ["FORCE_CUDA"] = "1"

    steps = [
        ("nvdiffrast",     ["pip", "install", "--no-cache-dir", "--no-build-isolation", "-e", "/app/nvdiffrast"]),
        ("diffoctreerast", ["pip", "install", "--no-cache-dir", "--no-build-isolation", "-e", "/app/diffoctreerast"]),
        ("diff-gaussian",  ["pip", "install", "--no-cache-dir", "--no-build-isolation",
                            "/app/mip-splatting/submodules/diff-gaussian-rasterization/"]),
    ]
    for name, cmd in steps:
        print(f"Compiling {name} ...", flush=True)
        result = subprocess.run(cmd, text=True, capture_output=True)
        # always print so we can see what nvcc actually did
        if result.stdout:
            print(result.stdout[-3000:], flush=True)
        if result.stderr:
            print(result.stderr[-3000:], flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"{name} compilation failed (exit {result.returncode})")

    # Editable installs register via .pth files processed only at Python startup.
    # The current process already started, so .pth files won't be picked up.
    # Manually insert source dirs so imports work immediately in this process.
    for src in ["/app/nvdiffrast", "/app/diffoctreerast"]:
        if src not in sys.path:
            sys.path.insert(0, src)

    os.makedirs("/workspace", exist_ok=True)
    open(_EXT_FLAG, "w").write(_IMAGE_VERSION)
    print("CUDA extensions compiled and cached.", flush=True)


def _needs_compilation() -> bool:
    if not os.path.exists(_EXT_FLAG):
        return True
    if open(_EXT_FLAG).read().strip() != _IMAGE_VERSION:
        return True
    if not _cuda_extensions_importable():
        return True
    return False


try:
    if _needs_compilation():
        print("Compiling CUDA extensions (~5 min on first run) ...", flush=True)
        _install_cuda_extensions()
    else:
        print("CUDA extensions already compiled (cached).", flush=True)

    print(f"Loading pipeline: {MODEL_ID} (cache: {HF_HOME})", flush=True)
    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(repo_id=MODEL_ID, cache_dir=HF_HOME)
    print(f"Model cached at: {local_model_dir}", flush=True)

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(local_model_dir)
    pipeline.cuda()
    print("Pipeline ready.", flush=True)

except Exception:
    _STARTUP_ERROR = traceback.format_exc()
    print(f"STARTUP FAILED:\n{_STARTUP_ERROR}", flush=True)
    try:
        os.makedirs("/workspace", exist_ok=True)
        with open("/workspace/startup_error.log", "w") as f:
            f.write(_STARTUP_ERROR)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")


def _fetch_image(url: str) -> Image.Image:
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGBA")


def load_image(job_input: dict) -> Image.Image:
    if "image" in job_input:
        return _decode_image(job_input["image"])
    if "image_url" in job_input:
        return _fetch_image(job_input["image_url"])
    raise ValueError("Input must contain 'image' (base64) or 'image_url'")


def load_images(job_input: dict) -> list:
    """Return a list of PIL images for multi-image mode."""
    if "images" in job_input:
        return [_decode_image(b64) for b64 in job_input["images"]]
    if "image_urls" in job_input:
        return [_fetch_image(url) for url in job_input["image_urls"]]
    return []


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def export_mesh(mesh: trimesh.Trimesh, fmt: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        path = tmp.name
    try:
        mesh.export(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    if _STARTUP_ERROR:
        return {"error": f"Worker startup failed: {_STARTUP_ERROR}"}

    inp = job.get("input", {})

    output_format = inp.get("output_format", "glb").lower()
    if output_format not in ("glb", "stl"):
        return {"error": f"Unsupported output_format '{output_format}'. Use 'glb' or 'stl'."}

    simplify_ratio = float(inp.get("simplify_ratio", 1.0))
    texture_size = int(inp.get("texture_size", 2048))
    seed = int(inp.get("seed", 42))
    sparse_steps = int(inp.get("sparse_steps", 50))
    sparse_cfg = float(inp.get("sparse_cfg", 7.5))
    slat_steps = int(inp.get("slat_steps", 50))
    slat_cfg = float(inp.get("slat_cfg", 7.5))

    multi_mode = inp.get("mode", "stochastic")
    if multi_mode not in ("stochastic", "multidiffusion"):
        return {"error": f"Unsupported mode '{multi_mode}'. Use 'stochastic' or 'multidiffusion'."}

    sampler_params = dict(
        seed=seed,
        sparse_structure_sampler_params={"steps": sparse_steps, "cfg_strength": sparse_cfg},
        slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
        formats=["gaussian", "mesh"],
        preprocess_image=True,
    )

    # Determine single vs. multi-image
    is_multi = "images" in inp or "image_urls" in inp

    try:
        if is_multi:
            images = load_images(inp)
            if len(images) < 2:
                return {"error": "Multi-image mode requires at least 2 images in 'images' or 'image_urls'."}
        else:
            images = [load_image(inp)]
    except Exception as e:
        return {"error": f"Image load failed: {e}"}

    try:
        if is_multi:
            outputs = pipeline.run_multi_image(
                images,
                mode=multi_mode,
                **sampler_params,
            )
        else:
            outputs = pipeline.run(
                images[0],
                **sampler_params,
            )

        if output_format == "stl":
            # For printing: use the raw FlexiCubes mesh directly.
            # to_glb decimates and bakes textures — wrong for print.
            raw = outputs["mesh"][0]
            verts = raw.vertices.cpu().float().numpy()
            faces = raw.faces.cpu().long().numpy()
            combined = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            combined = repair_mesh(combined)
            mesh_bytes = export_mesh(combined, "stl")
        else:
            glb_mesh = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify_ratio,
                texture_size=texture_size,
                verbose=False,
            )
            mesh_bytes = glb_mesh.export(file_type="glb")
            if isinstance(glb_mesh, trimesh.Scene):
                combined = trimesh.util.concatenate(list(glb_mesh.geometry.values()))
            else:
                combined = glb_mesh

        mesh_b64 = base64.b64encode(mesh_bytes).decode("utf-8")

        return {
            "mesh_b64": mesh_b64,
            "format": output_format,
            "vertex_count": int(len(combined.vertices)),
            "face_count": int(len(combined.faces)),
        }

    except Exception:
        return {"error": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
