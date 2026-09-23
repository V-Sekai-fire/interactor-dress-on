# Pixal3D image -> textured mesh (Stage 7) as a RunPod serverless (queue) worker.
# Built by RunPod's GitHub integration with the repository root as the build context:
#   Dockerfile path  tools/runpod/pixal3d/Dockerfile
# Nothing is compiled and no GPU is needed to build: python, torch and every CUDA
# extension are prebuilt packages pinned in tools/services/pixal3d/pixi.lock (its
# linux-64 half), the sources are the org forks fetch.py clones at pinned commits,
# and the weights come from the network volume at run time (MODELS_DIR).
# The extensions carry SASS for sm_80, sm_86 (so also sm_89), sm_100 and sm_120, and
# none for sm_90 (H100/H200): gates/7-pixal3d/linux-wheels. CUDA 13: driver >= 580.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# git: fetch.py clones the pinned sources, and pixi builds utils3d-moge from its git pin.
# gcc + libc6-dev: Triton compiles its CUDA launcher stub with the system C compiler at
# the first launch of each kernel (flex_gemm's sparse conv, FlexGEMM 2.0 in MoGe-3).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# pixi, the release the desk runs, checksum-pinned (sha256 of the GitHub release asset).
ARG PIXI_VERSION=0.79.0
ARG PIXI_SHA256=b9b6dd2bdf4e0043c2c0cd6d15334a26c6851121bf5ae16c39b69add598bafdc
RUN curl -fsSL -o /tmp/pixi.tar.gz \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
 && echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi \
 && rm /tmp/pixi.tar.gz \
 && pixi --version

WORKDIR /app

# The environment first (the big, slow layer): only the manifest and the lock.
COPY tools/services/svc_common.py tools/services/
COPY tools/services/pixal3d/pixi.toml tools/services/pixal3d/pixi.lock tools/services/pixal3d/
RUN cd tools/services/pixal3d \
 && pixi install --locked \
 && pixi clean cache --yes \
 && rm -rf /root/.cache

# The sources: fetch.py's source step only (clone the org forks at their pins, apply
# patches/); the weights are not fetched here.
COPY tools/services/pixal3d/fetch.py tools/services/pixal3d/serve.py tools/services/pixal3d/smoke.py tools/services/pixal3d/
COPY tools/services/pixal3d/patches/ tools/services/pixal3d/patches/
RUN cd tools/services/pixal3d && pixi run --locked python fetch.py --src
# fetch.py --weights (to fill a network volume from a pod) checks against the manifest.
COPY tools/models/manifest.tsv tools/models/

COPY tools/runpod/pixal3d/handler.py tools/runpod/pixal3d/

# NVIDIA_*: the container toolkit grants the driver (libcuda, nvidia-smi) to any image.
# CUDA_DEVICE_ORDER: nvidia-smi's numbering, as on the desk. The rest mirrors
# pixi.toml's [activation.env] plus serve.py's offline switch, because the worker runs
# the environment's python directly (no `pixi run` at start).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    MODELS_DIR=/runpod-volume/models \
    HF_HUB_OFFLINE=1 \
    ATTN_BACKEND=sdpa \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/tools/services/pixal3d/.pixi/envs/default/bin:$PATH

WORKDIR /app/tools/runpod/pixal3d
CMD ["python", "-u", "handler.py"]
