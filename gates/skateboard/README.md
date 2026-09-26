# Skateboard (issue 5): the character clip end to end, crude first

**Result: logic PASS headless; the render is CI's `gpu-macos` job.**
`project/skateboard.gd` poses the FoxGirl body on its 15-joint skeleton (Gate 8's
infer and rig fixtures) next to the draped garment, keys a 3-second wave on
one arm, and renders it at 1920 x 1080, 24 fps with Godot's movie writer
(CineForm `.cfhd` on Windows, Godot's AVI elsewhere).

Headless (`headless-logic.txt`, no RenderingDevice, so no pixels): 73 frames
posed, 0 non-finite vertices, 685 ms. The render runs on the macOS runner's
Metal GPU (`.github/workflows/build.yml`, step "Skateboard"), whose artifact
keeps `wave.avi` and `results.txt`.

Stand-ins, each printed in the results: the body and the rig are fixtures;
the skin weights are nearest-bone by distance, 69 vertices on the upper arm
and 546 on the forearm (a rig's weights replace them in the Bicycle); the
garment is rigid, it does not follow the arm. Rotations are 3x3 matrices from
an axis and an angle (rule 11).

Not yet: the wave keyed over MCP inside one Gate 8 run (the loop's fit takes
1393 s on the desk; this scene takes Gate 8's garment, or `--garment=` its
`<out>.fitted.obj`).

    godot --path project --rendering-driver vulkan --xr-mode off \
        --write-movie ../gates/skateboard/wave.avi --fixed-fps 24 --resolution 1920x1080 \
        --script skateboard.gd -- --out=res://../gates/skateboard/results.txt
