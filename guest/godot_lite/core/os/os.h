// godot-lite: OS reduced to what Cassie reads, the environment (its
// profiling and decimation switches).
#pragma once

#include "core/string/ustring.h"

#include <cstdint>

namespace gdl {

class OS {
public:
	static OS *get_singleton();

	bool has_environment(const String &p_var) const;
	String get_environment(const String &p_var) const;
};

} // namespace gdl
