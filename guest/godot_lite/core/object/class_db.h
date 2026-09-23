// godot-lite: the registration surface compiles and does nothing. There is
// no ClassDB in the guest: bind_method / bind_static_method swallow their
// arguments, the ADD_* / BIND_* macros expand to nothing, and GDCLASS only
// supplies get_class() and ends in `private:` as Godot's does.
#pragma once

#include "core/typedefs.h"

namespace gdl {

struct MethodDefinition {};

class ClassDB {
public:
	template <typename... Args>
	static void bind_method(Args &&...) {}
	template <typename... Args>
	static void bind_static_method(Args &&...) {}
};

} // namespace gdl

#define D_METHOD(...) ::gdl::MethodDefinition()
#define DEFVAL(m_defval) (m_defval)

#define ADD_PROPERTY(...)
#define ADD_SIGNAL(...)
#define ADD_GROUP(...)
#define BIND_ENUM_CONSTANT(...)
#define VARIANT_ENUM_CAST(...)

#define GDCLASS(m_class, m_inherits)                                  \
public:                                                               \
	using self_type = m_class;                                        \
	using super_type = m_inherits;                                    \
	static const char *get_class_static() { return #m_class; }        \
	virtual ::gdl::String get_class() const override { return #m_class; } \
                                                                      \
private:
