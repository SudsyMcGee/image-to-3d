FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_ID=JeffreyXiang/TRELLIS-image-large \
    HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone TRELLIS with submodules (includes flexicubes), plus CUDA extension repos
RUN git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git /app/TRELLIS
RUN git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /app/nvdiffrast
RUN git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /app/diffoctreerast
RUN git clone --depth 1 --recurse-submodules https://github.com/autonomousvision/mip-splatting.git /app/mip-splatting

# RunPod + huggingface + mesh utilities
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# TRELLIS Python-only deps — pin transformers for PyTorch 2.4 compatibility
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime open3d xatlas pyvista pymeshfix igraph \
    "transformers==4.44.2"

RUN pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir spconv-cu120

# kaolin — required by flexicubes submodule (pre-built wheel, no GPU needed at build time)
RUN pip install --no-cache-dir kaolin==0.17.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# flash-attn — pre-built wheel, no GPU or nvcc needed at build time
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
