// godot-lite: Resource is a RefCounted whose "changed" signal goes nowhere;
// loading, saving, naming and duplication are not modelled.
#pragma once

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"

namespace gdl {

class Resource : public RefCounted {
	GDCLASS(Resource, RefCounted);

public:
	void emit_changed() {}

	Resource() = default;
};

} // namespace gdl
