# Gate 0D — does stock Godot's OpenXR reach SteamVR?

**Result: the plumbing works; no head-mounted display was present.**

Two runs, both logged here:

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

## To turn this into a PASS

Either connect the headset so SteamVR presents an HMD, or enable SteamVR's
null driver for headset-less testing (`steamvr.vrsettings`:
`"driver_null": {"enable": true}`, `"steamvr": {"requireHmd": false,
"forcedDriver": "null"}`). The latter is what "test often" wants, and it is
the user's config to change.
