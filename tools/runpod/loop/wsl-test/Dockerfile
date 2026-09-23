# Desk test of the loop worker on the desk's own GPUs, before anything goes to RunPod.
#
# Docker Desktop runs Linux containers under WSL2, where Linux graphics reach the
# NVIDIA card through D3D12 (/dev/dxg, /usr/lib/wsl/lib: libd3d12 + NVIDIA's
# libnvwgf2umx). Mesa's dzn is Vulkan on top of that, on the RTX 3090 / 4090 (not
# the CPU); Ubuntu's Mesa does not build dzn, Arch ships it as vulkan-dzn. So this
# image is Arch + dzn + Xvfb around the loop image's own /opt/godot and /app,
# byte for byte. What it does not test is NVIDIA's Linux Vulkan driver, which only
# exists on a real Linux host (Gate 0H V2 ran it on RunPod).
#
#   docker build -f tools/runpod/loop/Dockerfile -t dress-on-loop:local .
#   docker build -f tools/runpod/loop/wsl-test/Dockerfile -t dress-on-loop:wsl-test .
#   MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v /usr/lib/wsl:/usr/lib/wsl:ro \
#       dress-on-loop:wsl-test python /app/handler.py --test_input '{"input": {}}'
FROM archlinux:latest
RUN pacman -Syu --noconfirm --needed vulkan-dzn vulkan-icd-loader xorg-server-xvfb libxcursor libxinerama \
        libxrandr libxi libxext libxrender libglvnd fontconfig libxkbcommon python python-pip \
    && pacman -Scc --noconfirm
RUN pip install --no-cache-dir --break-system-packages runpod==1.12.0
COPY --from=dress-on-loop:local /opt/godot /opt/godot
COPY --from=dress-on-loop:local /app /app
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json \
    PYTHONUNBUFFERED=1
CMD ["python", "-u", "/app/handler.py"]
