// godot-lite: core/typedefs.h includes "platform_config.h" first. As in
// Godot's platform configs, this is where alloca comes from (DynamicBVH
// uses it for its traversal stacks).
#pragma once

#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif
