"""
RunPod serverless handler: image (base64 or URL) → TRELLIS → GLB/STL (base64)

Input schema:
    {
        "image": "<base64-encoded PNG/JPG>",   # required (or use image_url)
        "image_url": "<https://...>",           # alternative to image
        "output_format": "glb" | "stl",        # default: "glb"
        "simplify_ratio": 0.95,                 # mesh decimation (0-1, default 0.95)
        "texture_size": 1024                    # texture resolution (default 1024)
    }

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
import base64
import io
import tempfile
import traceback

import runpod
import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, "/app/TRELLIS")
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# ---------------------------------------------------------------------------
# Model loaded once at worker startup, reused across requests
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "JeffreyXiang/TRELLIS-image-large")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")

print(f"Loading pipeline: {MODEL_ID} (cache: {HF_HOME})")
# Download weights on first cold start — they land on the network volume and are reused forever
from huggingface_hub import snapshot_download
import os as _os
snapshot_download(repo_id=MODEL_ID, cache_dir=HF_HOME)
pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID, cache_dir=HF_HOME)
pipeline.cuda()
print("Pipeline ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(job_input: dict) -> Image.Image:
    if "image" in job_input:
        raw = base64.b64decode(job_input["image"])
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    if "image_url" in job_input:
        import urllib.request
        with urllib.request.urlopen(job_input["image_url"]) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGBA")
    raise ValueError("Input must contain 'image' (base64) or 'image_url'")


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
    inp = job.get("input", {})

    output_format = inp.get("output_format", "glb").lower()
    if output_format not in ("glb", "stl"):
        return {"error": f"Unsupported output_format '{output_format}'. Use 'glb' or 'stl'."}

    simplify_ratio = float(inp.get("simplify_ratio", 0.95))
    texture_size = int(inp.get("texture_size", 1024))

    try:
        image = load_image(inp)
    except Exception as e:
        return {"error": f"Image load failed: {e}"}

    try:
        outputs = pipeline.run(
            image,
            seed=42,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )

        # Extract and post-process the mesh
        glb_bytes = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify_ratio,
            texture_size=texture_size,
            verbose=False,
        )

        if output_format == "glb":
            mesh_bytes = glb_bytes
        else:
            # Convert GLB → trimesh → STL
            scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
            if isinstance(scene, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(scene.geometry.values()))
            else:
                mesh = scene
            mesh = repair_mesh(mesh)
            mesh_bytes = export_mesh(mesh, "stl")

        mesh_b64 = base64.b64encode(mesh_bytes).decode("utf-8")

        # Quick stats via trimesh for the response
        quick_scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
        if isinstance(quick_scene, trimesh.Scene):
            combined = trimesh.util.concatenate(list(quick_scene.geometry.values()))
        else:
            combined = quick_scene

        return {
            "mesh_b64": mesh_b64,
            "format": output_format,
            "vertex_count": int(len(combined.vertices)),
            "face_count": int(len(combined.faces)),
        }

    except Exception:
        return {"error": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
