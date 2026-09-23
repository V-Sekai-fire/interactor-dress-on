# dress-on loop worker for RunPod serverless (queue): stock Godot 4.7.2 + the
# godot_sandbox addon + this repo's guest ELFs, run on the NVIDIA GPU through
# Vulkan on an Xvfb display (Gate 0H). Built by RunPod's GitHub integration with
# the repository root as the build context:
#   Dockerfile path  tools/runpod/loop/Dockerfile
# No GPU is needed to build; nothing is compiled here. fit.elf is gitignored (a
# 7 MB riscv64 build of cloth-fit that takes longer than RunPod's 30-minute build
# limit to cross-compile), so it comes from this repo's release (the tag is the
# root CITATION.cff version, v<version>), checksum-pinned.
# fit.elf from this repo's release; a desk build overrides the stage with a local
# copy (docker build --build-context fit-elf=<dir with fit.elf>); the checksum
# below holds either way.
FROM scratch AS fit-elf
ARG ELF_RELEASE=v0.1.0
ADD https://github.com/V-Sekai-fire/interactor-dress-on/releases/download/${ELF_RELEASE}/fit.elf /fit.elf

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb libvulkan1 mesa-vulkan-drivers libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 \
        libxext6 libxrender1 libgl1 libegl1 libfontconfig1 libxkbcommon0 \
        python3 python3-pip unzip wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir runpod==1.12.0

ARG GODOT_VERSION=4.7.2
RUN wget -q "https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}-stable/Godot_v${GODOT_VERSION}-stable_linux.x86_64.zip" \
    && unzip -q "Godot_v${GODOT_VERSION}-stable_linux.x86_64.zip" -d /opt/godot \
    && mv "/opt/godot/Godot_v${GODOT_VERSION}-stable_linux.x86_64" /opt/godot/godot \
    && rm "Godot_v${GODOT_VERSION}-stable_linux.x86_64.zip"

WORKDIR /app
COPY project/ /app/project/
COPY vendor/xr-grid/ /app/vendor/xr-grid/
ARG FIT_ELF_SHA256=f9f290c613eaf085b9f374d4b48d8210311508aede5cd689865c1caa3169bf27
COPY --from=fit-elf /fit.elf /app/project/fit.elf
RUN echo "$FIT_ELF_SHA256  /app/project/fit.elf" | sha256sum -c -
# The import registers the addon and the ELFs (first run after adding an ELF).
RUN /opt/godot/godot --headless --path /app/project --import > /tmp/import.log 2>&1 || (tail -20 /tmp/import.log; exit 1)

COPY tools/runpod/loop/handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
