# RunPod serverless: the whole flow as three queue endpoints

User, 2026-09-23: "setup serverless for our entire flow"; RunPod builds the images
from this GitHub repo; two endpoints (composition, AGENTS.md rule 6; a VoxHammer edit endpoint is a later stretch goal, after Pixal3D multi-view).

| endpoint | Dockerfile (build context = repo root) | what a job does | GPU | weights |
|---|---|---|---|---|
| `dress-on-loop` | `tools/runpod/loop/Dockerfile` | one run of the loop (Gate 8): stock Godot 4.7.2 + the guest ELFs on the NVIDIA GPU through Vulkan on Xvfb (Gate 0H) | any NVIDIA with Vulkan, ≥ 16 GB | none (ELFs + fixtures in the image) |
| `pixal3d` | `tools/runpod/pixal3d/Dockerfile` | image → textured mesh as an OpenUSD layer (`predict`, `extract`) | ≥ 24 GB (peak 18.3 GB) | network volume, `/runpod-volume/models` |

Both are **queue** endpoints, not load-balancing ones: a load-balanced request
is capped at 5.5 min, a Pixal3D job takes 4–5 min plus a cold start,
and one loop job takes ~11 min (the fit, with force_psd_projection). Jobs go to
`https://api.runpod.ai/v2/<endpoint id>/run` and are polled at
`/status/<job id>`; results are kept 30 min.

## What only the user can do (RunPod console, once)

1. **Settings → Connections → GitHub → Connect**, and give the RunPod app
   access to `V-Sekai-fire/interactor-dress-on`.
2. **Serverless → New Endpoint → Import Git Repository**, twice, choosing
   repository `V-Sekai-fire/interactor-dress-on`, branch `main`, endpoint type
   **Queue**, and the Dockerfile path from the table. Everything else (GPU list,
   workers 0–1, timeouts, volume, env) is then set through the REST API
   (`PATCH /v1/endpoints/<id>`).
3. RunPod rebuilds an endpoint only when a **GitHub release** is published; a
   push alone does not.

The API key never goes in the tree: it is read from the environment
(`RUNPOD_API_KEY`, the user's Windows user environment on the desk).

## Settings applied through the API

| | dress-on-loop | pixal3d |
|---|---|---|
| workers min/max | 0 / 1 | 0 / 1 |
| idle timeout | 5 s | 60 s |
| execution timeout | 3600 s | 900 s |
| FlashBoot | on | on |
| container disk | 20 GB | 40 GB |
| env | `NVIDIA_DRIVER_CAPABILITIES=all` (in the image) | `MODELS_DIR=/runpod-volume/models` |

## Job contracts

`dress-on-loop` input: `{"allow_fixture": "infer,rig", "wallclock": 3000}` —
stages not yet wired to a service stand in with their fixture only when
allowed, and the result says FIXTURE (as on the desk). Output: `result`
(`RESULT: PASS …`), `seconds`, `gpu`, the results lines, the JSON summary, the
fitted garment as OBJ text and the screenshot as base64 PNG; progress lines
stream through `/status`.

`pixal3d`: `{"route": "predict" | "extract" | "health", ...}` with
the same JSON body as the HTTP service (`tools/services/*/README.md`); the
worker runs the service's own server on loopback and forwards the job.
