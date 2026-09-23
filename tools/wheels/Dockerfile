# Build environment for the Pixal3D CUDA extension wheels (cp311, torch 2.8.0 + cu128,
# linux_x86_64). tools/wheels/build.sh builds this image and runs itself inside it;
# the build context is tools/wheels/. Compiling needs no GPU.
#
# Sources outside the org (AGENTS.md rule 1), all build tooling, none of it in a wheel
# beyond what the toolchain links:
#   nvidia/cuda:12.8.1-devel-ubuntu22.04 (ubuntu 22.04 + NVIDIA's CUDA 12.8 toolkit:
#     nvcc, cuobjdump, the libcuda stub), pinned by digest;
#   apt's git, curl, ca-certificates;
#   pixi 0.79.0 from prefix-dev/pixi's GitHub release, checked against its sha256;
#   conda-forge python/pip/setuptools/wheel/ninja and download.pytorch.org torch, as
#     pixi.lock pins them.
# The user accepted the ubuntu base, apt tooling and pixi as rule-1 exceptions
# (2026-09-23); the CUDA toolkit image is the same kind of exception (nvcc has no org fork).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

ARG PIXI_VERSION=0.79.0
ARG PIXI_SHA256=b9b6dd2bdf4e0043c2c0cd6d15334a26c6851121bf5ae16c39b69add598bafdc
RUN curl -fsSL -o /tmp/pixi.tar.gz \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
 && echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi \
 && rm /tmp/pixi.tar.gz

WORKDIR /env
COPY pixi.toml pixi.lock ./
RUN pixi install --locked && pixi clean cache --yes && rm -rf /root/.cache

ENV PATH=/env/.pixi/envs/default/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    WHEELS_IN_CONTAINER=1
WORKDIR /work
