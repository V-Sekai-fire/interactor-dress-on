# Pixal3D on RunPod serverless (queue worker)

Stage 7's image -> mesh service (tools/services/pixal3d, AGENTS.md's Stage 7
exception) as a RunPod serverless **queue** endpoint. The worker image runs the
service's own `serve.py` (the code path `pixi run serve` takes on the desk) on
127.0.0.1 inside the container; `handler.py` turns each RunPod job into one HTTP call
to it and returns the service's JSON. Nothing is compiled in the image and the build
needs no GPU: python, torch and the five CUDA extensions are prebuilt packages pinned
in `tools/services/pixal3d/pixi.lock` (its linux-64 half), the sources are the org
forks `fetch.py --src` clones at their pinned commits, and the ~27 GB of weights come
from a network volume at run time.

Files: `Dockerfile` (+ `Dockerfile.dockerignore`), `handler.py` (the worker),
`local_test.py` (the desk test below), `runs/` (its evidence; GLB/USD/PNG ignored).

Sources (AGENTS.md rule 1): every service dependency is an org fork or PyPI /
download.pytorch.org, as the lock records (tools/services/pixal3d). The image itself
also takes, as tooling: the public `ubuntu:24.04` base, apt's `git`, `gcc`,
`libc6-dev`, `curl`, `ca-certificates`, and pixi 0.79.0 (the desk's version) from
`prefix-dev/pixi`'s GitHub release, checked against its sha256; the org has no pixi
fork.

## Endpoint settings (RunPod console)

Nothing here has been created; these are the settings to enter.

| setting | value | why |
|---|---|---|
| Source | GitHub repo `V-Sekai-fire/interactor-dress-on`, the branch holding this directory | RunPod builds the image itself |
| Dockerfile path | `tools/runpod/pixal3d/Dockerfile` | build context is the repository root |
| Endpoint type | **Queue** (not load balancing) | jobs are `/run` / `/runsync` |
| GPUs | 24 GB or more, from: A100 (sm_80); RTX A5000, RTX 3090, A40, RTX A6000 (sm_86); L4, RTX 4090, L40, L40S, RTX 6000 Ada (sm_89); RTX 5090, RTX PRO 6000 (sm_120); B200 (sm_100). **Leave out every pool with H100 or H200** (sm_90) | the extension wheels carry SASS for sm_80/86/100/120 only, and sm_89 runs sm_86 code; nothing runs on sm_90 (gates/7-pixal3d/linux-wheels). Peak on a 3090: 14,967-15,213 MiB (nvidia-smi) |
| GPUs per worker | 1 | the handler gives the server one card (`CUDA_VISIBLE_DEVICES`, default `0`) |
| CUDA versions (advanced) | 13.0 and newer | torch 2.11.0+cu130 and the wheels link `libcudart.so.13`: host driver >= 580 |
| Active workers / max workers | 0 / 1 | scale to zero; one worker serves a predict and its extract |
| Idle timeout | 300 s | a cold start costs 3.3-4.3 min (below) and the first predict another ~2 min of Triton compiles; the default 5 s would put most extracts on a fresh worker |
| Execution timeout | 900 s | measured on a 3090: 187.5-198.8 s for the first predict on a worker, 86.0 s warm, 43.0-45.3 s for an extract; slower cards (L4) need headroom |
| FlashBoot | on | the cold start is model load (~27 GB read) plus Triton compiles |
| Container disk | 25 GB | the image is 10.24 GB; each extract leaves ~45 MB (GLB + USD) under /tmp; Triton and HF module caches |
| Network volume | attach one holding `models/` (below), >= 35 GB, in a data center with the GPUs above | RunPod mounts it at `/runpod-volume`; the image sets `MODELS_DIR=/runpod-volume/models` |
| Environment variables | none required | the image sets `MODELS_DIR`, `HF_HUB_OFFLINE=1`, `ATTN_BACKEND=sdpa`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Optional: `PIXAL3D_READY_TIMEOUT` (s, default 1200), `PIXAL3D_PORT` (default 18000), `MODELS_DIR` |

### The network volume

`/runpod-volume/models/` must hold the same tree as the desk's `models/`, every file
pinned by `tools/models/manifest.tsv` (32 files, 27.19 GB):
`Pixal3D-src/` (pipeline.json + ckpts/), `dinov3-vitl16/`, `moge-3-vitl/model.pt`,
`BiRefNet_HR-matting/`, `NAF/naf_release.pth`. Two ways to fill it:

- Upload the desk's copies with RunPod's S3-compatible API for network volumes (no
  compute needed): the five directories above, from `C:/interactor-dress-on/models/`.
- Or run this image once on a pod with the volume attached and fetch at the pinned
  revisions, checked against the manifest:
  `cd /app/tools/services/pixal3d && HF_HUB_OFFLINE=0 python fetch.py --weights`.
  Untested here. Note that `fetch.py` on this branch still downloads NAF from
  valeoai's release (main's manifest names the V-Sekai-fire/NAF mirror).

The worker only reads the volume. With `CUDA_VISIBLE_DEVICES` set by the handler,
svc_common's GPU round robin (a counter file written beside the weights) never runs,
so no worker writes to the shared volume.

## Job contract

`input.route` picks the service route; every other key is that route's request body,
unchanged (tools/services/pixal3d/README.md). The job's output is the route's JSON.

```json
{"input": {"route": "health"}}
{"input": {"route": "predict", "image": "<base64 PNG (RGBA or RGB), a data: URI, or an http(s) URL>",
           "seed": 42, "resolution": 1024, "nviews": 4, "fov": -1.0, "view_resolution": 512}}
{"input": {"route": "extract", "state": "<output.state of the predict>",
           "decimation_target": 210000, "texture_size": 2048}}
```

- `health` -> `{"status": "ok", "ready": true, "stub": false}`
- `predict` -> `{"state", "views": [base64 PNG], "cameras", "camera_params", "seed",
  "stub", "seconds", "vram_peak_mib"}`; `state` is the latent the client sends back.
- `extract` -> `{"glb": base64, "layer": base64 .usda, "stub", "seconds",
  "vram_peak_mib"}`; the layer's `/Asset/Geometry/Mesh_*` prims are UsdGeom.Mesh
  (points, faceVertexCounts/Indices, vertex normals, primvars:st; upAxis Y,
  metersPerUnit 1).
- Errors: a bad route, a 4xx/5xx from the service (e.g. `/predict: HTTP 400:
  resolution must be 512 or 1024, got 777`), or a server that did not start or died
  (`pixal3d service unavailable: ...`, plus `refresh_worker` so RunPod replaces the
  worker) come back as `{"error": ...}`, which RunPod reports as a FAILED job.

### Payload sizes (measured, 3090 run below)

| message | size | RunPod limit |
|---|---|---|
| predict request (17_img.png as base64) | 1.99 MB | 10 MB `/run`, 20 MB `/runsync` |
| predict response (state 4.41 MB + 4 views) | 4.91 MB | 20 MB |
| extract request (the state) | 4.41 MB | 10 MB / 20 MB |
| **extract response** (GLB 24.31 MB + USD layer **34.90 MB**, base64) | **59.21 MB** | **20 MB** |

The extract result is ~3x RunPod's result cap (RunPod's docs: 10 MB `/run`, 20 MB
`/runsync`; above ~20 MB the gateway refuses the worker's job-done call and the
client gets no output). The USD layer alone is 26.18 MB raw, 34.90 MB base64, for
202,134 points / 208,950 triangles. Not redesigned here: the options are returning the
USD only, compressing it (usdc), a lower `decimation_target`, or writing the result
to storage and returning a link.

## Local test (the desk, 2026-09-23)

```
docker build -f tools/runpod/pixal3d/Dockerfile -t pixal3d-runpod:dev .   # repo root
pixi run -m tools/services/pixal3d python tools/runpod/pixal3d/local_test.py --stub
pixi run -m tools/services/pixal3d python tools/runpod/pixal3d/local_test.py --gpu 0
```

`local_test.py` runs the image as RunPod would (CMD `handler.py`) with RunPod's local
API (`--rp_serve_api`), `docker run --gpus device=0` (the RTX 3090; the 4090 was
another agent's) and `C:/interactor-dress-on/models` mounted read-only at
`/runpod-volume/models`, then sends health, predict, extract through `POST /runsync`
and checks the USD as smoke.py does. The build was run from `git archive` of the
tree (the bytes a GitHub clone has), with `--no-cache`.

| run | result |
|---|---|
| build (Torch2110 wheels, the committed Dockerfile) | PASS, 420 s wall on the desk (16 CPUs, Docker Desktop), well inside RunPod's 30 min: apt 18 s, `pixi install --locked` 109 s (174 packages, 5.5 GiB of downloads, cache removed after), `fetch.py --src` 23 s, layer export + unpack 255 s. Image 10.24 GB (10,237,880,495 B), under the 80 GB cap; the context is 267 kB with `Dockerfile.dockerignore` |
| `smoke.py --imports-only` in the image, 3090 | PASS: torch 2.11.0+cu130, triton 3.6.0, all ten imports, nvdiffrast rasterizes 1352 px, flex_gemm Triton conv, CuMesh, na2d_block 8.9e-16 (gates/7-pixal3d/linux-wheels/torch2110-check-3090.log) |
| stub (`WEFTSPUN_STUB=1`, no GPU) | PASS in 8.6 s: worker up 5.1 s, health, predict, extract, layer opens (runs/stub.*) |
| `--test_input` health (stub) | PASS: one job, then the SDK exits and the server shuts down (atexit) |
| no GPU, no weights | FAILS LOUDLY as designed: serve.py exits 1 ("Found no NVIDIA driver"), the job returns `pixal3d service unavailable: serve.py exited with 1 before /health was ready` + refresh_worker |
| real, 3090 (runs/real.*) | **PASS in 436.1 s.** Worker up 198.8 s after `docker run` (server ready 195.3 s). predict 187.5 s (server 187.4 s, torch peak 13,231 MiB; camera_angle_x 0.7271, distance 1.3142 as on the desk). extract 43.0 s (server 40.5 s, peak 6,322 MiB). USD: 1 UsdGeom.Mesh, 202,134 points, 208,950 triangles, vertex normals, upAxis Y, metersPerUnit 1. nvidia-smi peak 14,967 MiB (33 before) |
| real, two predicts on one worker (runs/real-warm.*) | **PASS in 592.8 s.** Worker up 255.0 s. predict #1 198.8 s, **predict #2 86.0 s** (same worker, same input): the first predict on a fresh worker pays ~113 s of one-time work, most likely Triton JIT + autotune (flex_gemm's conv, FlexGEMM 2.0 in MoGe-3; both caches start empty in every container; not instrumented). extract 45.3 s (peak 6,405 MiB). USD: 201,046 points, 206,938 triangles. nvidia-smi peak 15,213 MiB |

Compared with the desk's native win-64 run (tools/services/pixal3d/runs, torch 2.8.0):
model load 130.1 s -> 195.3 s here (the weights were read through Docker Desktop's
Windows bind mount; a RunPod network volume is also network storage), predict
103.2 s -> 187.5 s, extract 43.3 s -> 43.0 s, nvidia-smi peak 18,285 -> 14,967 MiB
(`expandable_segments` takes effect on Linux; Windows ignores it). A warm predict is
86.0 s, faster than the desk's 103.2 s. The mesh is not bit-reproducible on the same
seed: 202,134 / 208,950 and 201,046 / 206,938 points / triangles in the two container
runs, 195,434 / 198,046 on the desk.

So a job's latency on a cold worker is ~3-4 min of start (model load) plus ~3.3 min
for the first predict; later predicts on the same worker take ~1.5 min. The Triton
caches (`~/.triton`, `~/.flex_gemm2`) are per container and not kept anywhere; pointing
`TRITON_CACHE_DIR` at the volume would carry them across workers (not done: concurrent
workers would write to the shared volume).

Negative results (kept):

- The plan's `wheels/Linux/Torch291` set (torch 2.9.1, cp312) built (598 s, 13.87 GB)
  and passed every import, then failed on the 3090: `nvdiffrast: Cuda error: 209`
  and `flex_gemm conv: no kernel image is available for execution on the device`.
  Its flex_gemm, nvdiffrast, nvdiffrec_render and o_voxel carry sm_120 code only
  (gates/7-pixal3d/linux-wheels). Its o_voxel also names
  `cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git` and the FlexGEMM
  equivalent as dependencies, which uv refuses beside the org URLs and which point
  outside the org; the Torch2110 o_voxel names neither.
- Not measured: sm_89 (4090, L4, L40S), sm_100, sm_120, and RunPod itself (no
  endpoint, template or volume was created).
