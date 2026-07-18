FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_ID=JeffreyXiang/TRELLIS-image-large \
    HF_HOME=/workspace/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone TRELLIS and install its build dependencies first
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /app/TRELLIS

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install TRELLIS with all extras (compiles CUDA extensions)
RUN cd /app/TRELLIS && pip install --no-cache-dir -e ".[all]"

# Weights are NOT baked in — they live on a RunPod network volume at /workspace/hf_cache
# Handler downloads them on first cold start, then they're cached on the volume forever

COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
