// godot-lite: Time's monotonic ticks (steady_clock). Inside the sandbox the
// guest clock is not a trustworthy clock (AGENTS.md); these ticks are for
// Cassie's opt-in profiling prints only.
#pragma once

#include "core/object/object.h"

#include <cstdint>

namespace gdl {

class Time : public Object {
	GDCLASS(Time, Object);

public:
	static Time *get_singleton();

	uint64_t get_ticks_usec() const;
};

} // namespace gdl
