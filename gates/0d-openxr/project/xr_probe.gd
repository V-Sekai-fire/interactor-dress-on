# GATE 0D: does stock Godot's OpenXR reach the SteamVR runtime on this machine?
#
# Wall-clock quit via a SceneTree timer: with --xr-mode on the frame loop can
# stall while the runtime hands over, which made --quit-after never fire. And
# every branch quits -- the first version left the "present but not
# initialized" case running until `timeout` killed it (exit 124), which reads
# as a hang rather than the clear FAIL it was.
extends SceneTree

func _init() -> void:
	create_timer(10.0).timeout.connect(func(): print("(10 s wall clock)"); quit(2))
	var xr = XRServer.find_interface("OpenXR")
	print("openxr interface: ", xr)
	if xr == null:
		print("FAIL: no OpenXR interface (openxr/enabled off, or no runtime)")
		quit(1)
		return
	print("initialized: ", xr.is_initialized())
	if xr.is_initialized():
		print("PASS: OpenXR session up")
		quit(0)
	else:
		print("FAIL: OpenXR present but not initialized -- runtime not reached")
		quit(1)
