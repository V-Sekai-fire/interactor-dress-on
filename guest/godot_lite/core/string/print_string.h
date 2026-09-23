// godot-lite: print_line / print_error to stdout / stderr, and WARN_VERBOSE
// (verbose output is off).
#pragma once

#include "core/string/ustring.h"
#include "core/variant/variant.h"

namespace gdl {

inline bool is_print_verbose_enabled() {
	return false;
}

void __print_line(const String &p_string);
String stringify_variants(const Array &p_args);

template <typename... Args>
void print_line(const Variant &p_variant, Args... p_args) {
	Array args;
	args.push_back(p_variant);
	(args.push_back(Variant(p_args)), ...);
	__print_line(stringify_variants(args));
}

} // namespace gdl

#define WARN_VERBOSE(m_msg)                   \
	{                                         \
		if (::gdl::is_print_verbose_enabled()) { \
			WARN_PRINT(m_msg);                \
		}                                     \
	}
