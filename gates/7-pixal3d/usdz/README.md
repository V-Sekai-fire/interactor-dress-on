# Pixal3D /extract as one .usdz: does it fit RunPod, and does the check see faults?

Question: RunPod carries at most 20 MB of job result. The Pixal3D service's /extract
answered with a GLB plus a text .usda layer, 59.21 MB of base64 for one asset
(tools/runpod/pixal3d/README.md). Does one .usdz (a USDC layer, the mesh bound to a
UsdPreviewSurface, the GLB's textures as PNG inside) fit under 18 MB of base64, what
does it cost, and does the smoke check (`tools/services/pixal3d/smoke.py`
`check_usdz`) actually catch a broken package?

## Controls (no GPU)

```
pixi run -m tools/services/pixal3d python gates/7-pixal3d/usdz/controls.py > gates/7-pixal3d/usdz/controls.log
pixi run -m tools/services/pixal3d python gates/7-pixal3d/usdz/handler_cap.py > gates/7-pixal3d/usdz/handler_cap.log 2>&1
```

`controls.py` builds the stub asset with the server's own writer (`_stub_glb` ->
`_to_usdz`, a textured quad) and checks it, then breaks one thing at a time.
`controls.log`, 2026-09-23: **PASS**, every case caught for its own reason:

| package | check_usdz says |
|---|---|
| the server's | 0 problems, UsdValidation silent |
| same files zipped deflated (Python `zipfile`) | compressed + unaligned for all 3 files; the layer is not a crate; the stage does not open ("compressed files are not supported") |
| stored but shifted off 64-byte alignment | unaligned for all 3; UsdValidation `ByteMisalignment` errors |
| metallicRoughness PNG left out | does not resolve inside the package; UsdValidation `UnresolvableDependency` |
| material binding removed | "no material bound"; UsdValidation `InvalidMaterialCollection` |
| `.usda` as the root layer | "first file in the package is not a .usdc layer" |
| a 1000 B budget | over budget |

`handler_cap.py` runs the RunPod worker's `handler()` on the desk (a stand-in `runpod`
module; the real `serve.py` in stub mode): under the default 20,000,000 B cap an extract
job comes back as the service's JSON (`format` usdz, `usd_b64` 4,784 B); with the cap at
1,000 B the same job comes back as `{"error": "/extract: result 5100 B is over RunPod's
1000 B result cap ..."}`, a failed job rather than a result the gateway would drop.
`handler_cap.log`: **PASS**.

## Measured (native win-64, RTX 4090)

The evidence is tools/services/pixal3d/runs/real-sweep.* (one predict, extracts at
texture_size 2048, 1536, 1024 on the same state) and runs/real.* (`pixi run smoke`
at the new default); the tables are in tools/services/pixal3d/README.md, "USDZ
result". In short:

| texture_size | .usdz | base64 | layer / textures |
|---|---|---|---|
| 2048 | 14.86 MB | 19.81 MB, **over 18 MB** | 47% / 53% |
| 1536 (default now) | 11.70 MB (11.14 MB in the default run) | 15.60 MB (14.85 MB) | 60% / 40% (59% / 41%) |
| 1024 | 9.24 MB | 12.32 MB | 76% / 24% |

`recheck.py` (`recheck.log`) runs the final check_usdz + compare_glb over those four
packages and the GLBs the server kept: every one has points, faces and st identical
to its GLB, st equal to (u, 1 - v) of the GLB's raw TEXCOORD_0 bytes (glTF's uv origin
is the image's top left, USD's st origin its bottom left; so the textures are not
flipped), identical RGB texels in both textures, no UsdValidation finding; the only
problem is 2048's budget.

The USDC layer (~6.6-7.0 MB for 186-199k points) is the floor: points, normals and st
are float arrays the crate stores as they are. So the texture size, not the container
format, decides whether a result fits; 1536 is the largest tried that does, with 2.4 MB
of base64 to spare at 199k points. Not measured: other input images (texture entropy
moves the PNG sizes), and the result through a real RunPod endpoint (none exists).
