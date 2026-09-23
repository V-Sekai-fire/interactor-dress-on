// godot-lite: the CoreGlobals flags the vendored templates read.
#pragma once

namespace gdl {

class CoreGlobals {
public:
	static inline bool leak_reporting_enabled = true;
	static inline bool print_line_enabled = true;
	static inline bool print_error_enabled = true;
};

} // namespace gdl
