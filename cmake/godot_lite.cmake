# godot_lite: the namespaced Godot shim Cassie compiles against (Cut 4).
# guest/godot_lite is hand-written (Vector, Variant, Object, ...);
# vendor/godot-core-subset is Godot's own math/templates/Curve3D, extracted
# by tools/vendor/godot_core_subset.py. Everything is in namespace gdl; the
# sources need no prelude. Consumers that want Godot's unqualified names
# force-include guest/godot_lite/gdl_prelude.h themselves (cassie_core does).
#
# EXCLUDE_FROM_ALL like the rest of the curvenet stage: built when a target
# links it. tests/native/godot_lite/build.sh builds the same sources natively.

get_filename_component(GODOT_LITE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

file(GLOB_RECURSE GODOT_LITE_SOURCES
	"${GODOT_LITE_ROOT}/guest/godot_lite/*.cpp"
	"${GODOT_LITE_ROOT}/vendor/godot-core-subset/*.cpp"
)

add_library(godot_lite STATIC EXCLUDE_FROM_ALL ${GODOT_LITE_SOURCES})
target_include_directories(godot_lite PUBLIC
	"${GODOT_LITE_ROOT}/guest/godot_lite"
	"${GODOT_LITE_ROOT}/vendor/godot-core-subset"
)
target_compile_features(godot_lite PUBLIC cxx_std_17)
