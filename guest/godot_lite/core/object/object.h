// godot-lite: Object without ClassDB, signals, metadata or ObjectDB.
// emit_signal and the property-list notifications are no-ops; get_class()
// comes from GDCLASS; cast_to is dynamic_cast.
#pragma once

#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "core/variant/callable.h"
#include "core/variant/variant.h"

// Godot's object.h brings these in transitively; Cassie relies on that.
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/templates/local_vector.h"

namespace gdl {

class Object {
public:
	using self_type = Object;
	static const char *get_class_static() { return "Object"; }
	virtual String get_class() const { return "Object"; }

	template <typename T>
	static T *cast_to(Object *p_object) {
		return dynamic_cast<T *>(p_object);
	}
	template <typename T>
	static const T *cast_to(const Object *p_object) {
		return dynamic_cast<const T *>(p_object);
	}

	// Signals are not connected to anything in the guest.
	template <typename... VarArgs>
	Error emit_signal(const StringName &, VarArgs...) { return OK; }
	void notify_property_list_changed() {}

	// Method dispatch by name for Callable(object, "method"). Without ClassDB
	// nothing is registered: override to expose methods.
	virtual Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	virtual bool is_ref_counted() const { return false; }

	Object() = default;
	Object(const Object &) = delete;
	Object &operator=(const Object &) = delete;
	virtual ~Object() = default;
};

} // namespace gdl
