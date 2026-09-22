# Gate 0D — does stock Godot's OpenXR reach a runtime with an HMD?

**Result: PASS on OpenXR-Simulator.** `run-simulator.log`:

```
OpenXR: Created instance for OpenXR 1.1.54
OpenXR: Running on OpenXR runtime:  OpenXR Simulator Runtime   1.0.27
OpenXR: XrGraphicsRequirementsVulkan2KHR:
PASS: OpenXR session up
```

Stock Godot 4.7.2, `--rendering-driver vulkan --xr-mode on`, and
`XR_RUNTIME_JSON` pointed at `tools/openxr-simulator/openxr_simulator.abs.json`
(elliotttate/OpenXR-Simulator 1.5.0, the user's choice over SteamVR's null
driver). Nothing registered system-wide; SteamVR untouched. The simulator
opens a desktop stereo window, so from here every VR stage can be exercised
without a headset.

One trap cost a run: the zip's own `openxr_simulator.json` says
`"library_path": "openxr_simulator.dll"`, a bare filename, and the OpenXR
loader searches a bare filename on the *system* library path, not beside the
manifest — "failed to load with error 2". The `.abs.json` beside it carries an
absolute path with forward slashes (a backslash-escaped one fails the loader's
JSON parse). `tools/openxr-simulator/` is gitignored; fetch 1.5.0 from the
release and write the `.abs.json` as the README there describes.

The two headset runtimes, kept as the controls (both logged here):

- `run.log` — default runtime. `OpenXR: Running on OpenXR runtime:
  VirtualDesktopXR 1.0.10`, then `XR_ERROR_FORM_FACTOR_UNAVAILABLE`. The
  machine's **active** OpenXR runtime is Virtual Desktop's, not SteamVR, and it
  saw no headset.
- `run-steamvr.log` — with `XR_RUNTIME_JSON` pointed at
  `SteamVR\steamxr_win64.json`. `OpenXR: Created instance for OpenXR 1.0.54`,
  `Running on OpenXR runtime: SteamVR/OpenXR 2.17.10`, then the **same**
  `XR_ERROR_FORM_FACTOR_UNAVAILABLE`.

So stock Godot creates the instance, selects the runtime, enumerates its
extensions, and honours a per-process runtime override. What neither runtime
had at the time was an HMD to open a session on. `vrserver.exe` running is
not the same as a headset being connected.

## How VR gates run here

- Always `--rendering-driver vulkan --xr-mode on`, never `--headless`.
- `XR_RUNTIME_JSON=<SteamVR json>` per process. The system default is the
  user's; it is not flipped from a build script.
- Quit on a wall clock (`SceneTree.create_timer(10.0)`), and quit in **every**
  branch. The first version used `--quit-after`, which never fires while the
  XR frame loop is stalled on runtime handover, and its not-initialized branch
  fell through without quitting — both read as a 60 s hang rather than the
  clear FAIL they were.
- Pair every VR run with a flat control
  (`0a-renderingdevice/project/control.gd` is the template) so "XR blocked
  it" is separable from "no runtime was present".

## With a real headset

Connect it so SteamVR (or VirtualDesktopXR) presents an HMD and run the same
script with `XR_RUNTIME_JSON` at that runtime's manifest. The simulator run is
the one the stages are tested against day to day.
