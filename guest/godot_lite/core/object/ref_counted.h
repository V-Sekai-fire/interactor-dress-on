// godot-lite: RefCounted and Ref<T>, Godot's intrusive reference counting.
// A fresh object has count 0; the first Ref (or Variant) takes it to 1, and
// the last one to let go deletes it.
#pragma once

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/os/memory.h"
#include "core/templates/hashfuncs.h"

#include <atomic>

namespace gdl {

class RefCounted : public Object {
	GDCLASS(RefCounted, Object);
	std::atomic<int64_t> refcount{ 0 };

public:
	bool is_referenced() const { return refcount.load() > 0; }
	bool init_ref() { return reference(); }
	bool reference() {
		refcount.fetch_add(1);
		return true;
	}
	// True when the count dropped to zero: the caller deletes.
	bool unreference() { return refcount.fetch_sub(1) == 1; }
	int get_reference_count() const { return int(refcount.load()); }
	bool is_ref_counted() const override { return true; }

	RefCounted() = default;
	~RefCounted() override = default;
};

template <typename T>
class Ref {
	T *reference = nullptr;

	void ref_pointer(T *p_refcounted) {
		if (p_refcounted == reference) {
			return;
		}
		Ref cleanup_ref;
		cleanup_ref.reference = reference;
		reference = p_refcounted;
		if (reference) {
			reinterpret_cast<RefCounted *>(reference)->reference();
		}
	}

public:
	static String get_class_static() { return T::get_class_static(); }

	bool operator==(const T *p_ptr) const { return reference == p_ptr; }
	bool operator!=(const T *p_ptr) const { return reference != p_ptr; }
	bool operator<(const Ref<T> &p_r) const { return reference < p_r.reference; }
	bool operator==(const Ref<T> &p_r) const { return reference == p_r.reference; }
	bool operator!=(const Ref<T> &p_r) const { return reference != p_r.reference; }

	T *operator*() const { return reference; }
	T *operator->() const { return reference; }
	T *ptr() const { return reference; }

	operator Variant() const { return Variant(static_cast<const Object *>(reference)); }

	void operator=(const Ref &p_from) { ref_pointer(p_from.reference); }
	void operator=(Ref &&p_from) {
		if (reference == p_from.reference) {
			return;
		}
		unref();
		reference = p_from.reference;
		p_from.reference = nullptr;
	}
	template <typename T_Other>
	void operator=(const Ref<T_Other> &p_from) {
		ref_pointer(Object::cast_to<T>(p_from.ptr()));
	}
	void operator=(T *p_from) { ref_pointer(p_from); }
	void operator=(const Variant &p_variant) {
		Object *object = p_variant.get_validated_object();
		if (object == reference) {
			return;
		}
		ref_pointer(Object::cast_to<T>(object));
	}
	template <typename T_Other>
	void reference_ptr(T_Other *p_ptr) {
		if (reference == p_ptr) {
			return;
		}
		ref_pointer(Object::cast_to<T>(p_ptr));
	}

	Ref(const Ref &p_from) { this->operator=(p_from); }
	Ref(Ref &&p_from) {
		reference = p_from.reference;
		p_from.reference = nullptr;
	}
	template <typename T_Other>
	Ref(const Ref<T_Other> &p_from) { this->operator=(p_from); }
	Ref(T *p_from) { this->operator=(p_from); }
	Ref(const Variant &p_from) { this->operator=(p_from); }

	bool is_valid() const { return reference != nullptr; }
	bool is_null() const { return reference == nullptr; }

	void unref() {
		if (reference) {
			// T has single inheritance down to RefCounted (as Godot assumes).
			if (reinterpret_cast<RefCounted *>(reference)->unreference()) {
				memdelete(reinterpret_cast<RefCounted *>(reference));
			}
			reference = nullptr;
		}
	}

	template <typename... VarArgs>
	void instantiate(VarArgs... p_params) {
		Ref<T> ref = memnew(T(p_params...));
		SWAP(reference, ref.reference);
	}

	uint32_t hash() const { return HashMapHasherDefault::hash(reference); }

	Ref() = default;
	~Ref() { unref(); }
};

} // namespace gdl
