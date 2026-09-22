# Control for Gate 0A: can GDScript itself get a local RenderingDevice here?
# If the host cannot, the guest's failure says nothing about the sandbox.
extends SceneTree

func _init() -> void:
	var rd = RenderingServer.create_local_rendering_device()
	print("host RenderingDevice: ", rd)
	quit(0 if rd != null else 1)
