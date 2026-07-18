"""
Runs at Docker build time to bake model weights into the image.
With a RunPod network volume, set SKIP_BAKE=1 and mount weights at /workspace/hf_cache instead.
"""
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "JeffreyXiang/TRELLIS-image-large")
cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")

if os.environ.get("SKIP_BAKE"):
    print(f"SKIP_BAKE set — skipping weight download (expected on network volume at {cache_dir})")
else:
    print(f"Downloading {model_id} to {cache_dir} ...")
    snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print("Done.")
