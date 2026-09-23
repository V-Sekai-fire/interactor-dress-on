// godot-lite prelude: force-included (-include gdl_prelude.h) into every
// Cassie TU so its unqualified Godot names (Vector3, Ref, Dictionary, MAX...)
// resolve into namespace gdl without editing the sources.
#pragma once

namespace gdl {}
using namespace gdl;
