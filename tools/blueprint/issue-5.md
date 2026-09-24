The "GPT-6 Astra 3D Character Workflow: Rig and Animate from One Image" mapped onto this repo's loop (image → curvenet → dress-on → drape, as godot-sandbox guest ELFs over RenderingDevice), as of main 5187fcb and 2026-09-24.

| Step | What they use | What we have | State |
| :---- | :---- | :---- | :---- |
| Reference image | a front-facing T-pose image | several views of the character, for Pixal3D's multi-view model | ready |
| Split the reference into body, head and hair | an image-editing pass per part | an instruction-following generative model, as a guest ELF on ggml-rd, applied to each view; MaskScore / EditScore score each edit | decided; deferred to a later session |
| Body, head and hair as separate 3D parts | Tripo image-to-3D, run once per edited reference | Pixal3D image-to-textured-mesh service, multi-view only (its `_mv` checkpoints), returning OpenUSD (Stage 7) | in progress: op lists and tensor schemas in `gates/7-pixal3d`, no gate result yet; Gate 8 runs INFER from a fixture; the USD import passed Gate 0G |
| Assistant drives the editor | Blender MCP | transport-godot-mcp (un-archived 2026-09-24): every guest entry point callable over MCP | done: Gate 0E PASS, Gate 8's MCP drive ends `DONE` |
| Assemble head and hair onto the body | the assistant aligns and closes gaps | curvenet, the fit (PolyFEM and AVBD), the intersection check, the similarity fit | done for garments (Gate 8 PASS on FoxGirl: fit 1393 s, `OK none`, 100 drape steps on rd); not yet tried on hair or a head |
| Body model | the image-to-3D body | ANNY forward and backward kernels in Lean, gradients within 1.7e-6 of finite differences (#4) | kernels only; `anny.elf` on real ANNY data against AnnyInverter's 2.458 / 2.281 mm is deferred |
| Rig and skin | Tripo's auto-rigged GLB, then weight fixes | SkinTokens/TokenRig on ggml-rd (Stage 4b), mapped onto our 15-joint skeleton | the decode step's matmuls pass Gate 3's L2 test (9.2 ms GPU per step); the full stage is not started, and Gate 8 takes the rig from a fixture |
| Materials | fix up the imported maps | USD's preview-surface material into Godot | small |
| 7a. Animation from existing motion | import a dance | Kimodo text-to-motion or MotionBricks, both on ggml-rd | MotionBricks' ops are all in ggml-rd's 105 kernels; Kimodo's denoiser and text layer are oracle graphs with no gate log yet |
| 7b. Keyframed greeting | the assistant keys a 3-second wave | the same thing over Godot MCP | ready |
| Secondary motion | keyed hair | the AVBD drape: cloth now, hair as the same solver | Gate 5: G10 and six others pass. G2 is next: x and g in df32, judged on optimality, LBFGSpp as the control. G9 is next: calibrate cpu vs rd at open. G5 is deferred: qf32 AVBD state. Hair not tried |
| Facial expressions | continuous face controls | ANNY/MakeHuman expression targets through the ANNY blend kernels (#4) | decided; deferred |
| Render at 1920 × 1080, 24 fps | Blender | Godot with the vendored CineForm MovieWriter | ready |
| CI | — | Linux build and headless gates; GPU gates on the macOS runner's Metal device (#6) | main red on the no-GPU `rd_probe` check; the fix is in #6, #4 and #3, awaiting their runs |
| Org rules | — | RunPod mentions, CPU-execution exemption, pixi root, no C++ `auto` or env-var config, manifest placement at `3-interactor/dress-on` | deferred |
