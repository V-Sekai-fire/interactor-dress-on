#include "core/variant/variant.h"

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/templates/hashfuncs.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"

#include <cmath>
#include <cstdio>

namespace gdl {

// --- construction -----------------------------------------------------------

Variant::Variant(bool p_bool) {
	type = BOOL;
	_data._bool = p_bool;
}

#define GDL_VARIANT_INT_CTOR(m_type) \
	Variant::Variant(m_type p_int) { \
		type = INT;                  \
		_data._int = int64_t(p_int); \
	}
GDL_VARIANT_INT_CTOR(signed char)
GDL_VARIANT_INT_CTOR(unsigned char)
GDL_VARIANT_INT_CTOR(short)
GDL_VARIANT_INT_CTOR(unsigned short)
GDL_VARIANT_INT_CTOR(int)
GDL_VARIANT_INT_CTOR(unsigned int)
GDL_VARIANT_INT_CTOR(long)
GDL_VARIANT_INT_CTOR(unsigned long)
GDL_VARIANT_INT_CTOR(long long)
GDL_VARIANT_INT_CTOR(unsigned long long)
#undef GDL_VARIANT_INT_CTOR

Variant::Variant(float p_float) {
	type = FLOAT;
	_data._float = p_float;
}

Variant::Variant(double p_float) {
	type = FLOAT;
	_data._float = p_float;
}

Variant::Variant(const String &p_string) { _set_payload(STRING, p_string); }
Variant::Variant(const char *p_string) { _set_payload(STRING, String(p_string)); }
Variant::Variant(const Vector3 &p_v) { _set_payload(VECTOR3, p_v); }
Variant::Variant(const Callable &p_callable) { _set_payload(CALLABLE, p_callable); }
Variant::Variant(const Dictionary &p_dictionary) { _set_payload(DICTIONARY, p_dictionary); }
Variant::Variant(const Array &p_array) { _set_payload(ARRAY, p_array); }
Variant::Variant(const PackedInt32Array &p_v) { _set_payload(PACKED_INT32_ARRAY, p_v); }
Variant::Variant(const PackedFloat32Array &p_v) { _set_payload(PACKED_FLOAT32_ARRAY, p_v); }
Variant::Variant(const PackedVector3Array &p_v) { _set_payload(PACKED_VECTOR3_ARRAY, p_v); }

Variant::Variant(const Object *p_object) {
	type = OBJECT;
	Object *obj = const_cast<Object *>(p_object);
	_data._obj = obj;
	if (obj && obj->is_ref_counted()) {
		RefCounted *rc = static_cast<RefCounted *>(obj);
		rc->reference();
		// The shared_ptr owns exactly one reference for every copy of this Variant.
		_ptr = std::shared_ptr<const void>(static_cast<const void *>(rc), [](const void *p_rc) {
			RefCounted *r = static_cast<RefCounted *>(const_cast<void *>(p_rc));
			if (r->unreference()) {
				memdelete(r);
			}
		});
	}
}

// --- conversion -------------------------------------------------------------

Variant::operator bool() const {
	switch (type) {
		case BOOL:
			return _data._bool;
		case INT:
			return _data._int != 0;
		case FLOAT:
			return _data._float != 0.0;
		case STRING:
			return !_payload<String>()->is_empty();
		case VECTOR3:
			return *_payload<Vector3>() != Vector3();
		case OBJECT:
			return _data._obj != nullptr;
		case CALLABLE:
			return !_payload<Callable>()->is_null();
		case DICTIONARY:
			return !_payload<Dictionary>()->is_empty();
		case ARRAY:
			return !_payload<Array>()->is_empty();
		case PACKED_INT32_ARRAY:
			return !_payload<PackedInt32Array>()->is_empty();
		case PACKED_FLOAT32_ARRAY:
			return !_payload<PackedFloat32Array>()->is_empty();
		case PACKED_VECTOR3_ARRAY:
			return !_payload<PackedVector3Array>()->is_empty();
		default:
			return false;
	}
}

Variant::operator long long() const {
	switch (type) {
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return _data._int;
		case FLOAT:
			return (long long)_data._float;
		case STRING:
			return _payload<String>()->to_int();
		default:
			return 0;
	}
}

Variant::operator double() const {
	switch (type) {
		case BOOL:
			return _data._bool ? 1.0 : 0.0;
		case INT:
			return double(_data._int);
		case FLOAT:
			return _data._float;
		case STRING:
			return _payload<String>()->to_float();
		default:
			return 0.0;
	}
}

Variant::operator String() const {
	return type == STRING ? *_payload<String>() : stringify();
}

Variant::operator Vector3() const {
	return type == VECTOR3 ? *_payload<Vector3>() : Vector3();
}

Variant::operator Callable() const {
	return type == CALLABLE ? *_payload<Callable>() : Callable();
}

Variant::operator Dictionary() const {
	return type == DICTIONARY ? *_payload<Dictionary>() : Dictionary();
}

template <typename T>
static Array _packed_to_array(const Vector<T> &p_packed) {
	Array a;
	a.resize(p_packed.size());
	for (int64_t i = 0; i < p_packed.size(); i++) {
		a[i] = Variant(p_packed[i]);
	}
	return a;
}

Variant::operator Array() const {
	switch (type) {
		case ARRAY:
			return *_payload<Array>();
		case PACKED_INT32_ARRAY:
			return _packed_to_array(*_payload<PackedInt32Array>());
		case PACKED_FLOAT32_ARRAY:
			return _packed_to_array(*_payload<PackedFloat32Array>());
		case PACKED_VECTOR3_ARRAY:
			return _packed_to_array(*_payload<PackedVector3Array>());
		default:
			return Array();
	}
}

// Same packed type: share the buffer (COW). Anything array-like: convert
// element by element, as Godot's Variant does.
template <typename T>
static Vector<T> _to_packed(const Variant &p_v, Variant::Type p_type, const std::shared_ptr<const void> &p_ptr) {
	if (p_v.get_type() == p_type) {
		return *static_cast<const Vector<T> *>(p_ptr.get());
	}
	const Array a = p_v.operator Array();
	Vector<T> out;
	out.resize(a.size());
	T *w = out.ptrw();
	for (int i = 0; i < a.size(); i++) {
		w[i] = static_cast<T>(a[i]);
	}
	return out;
}

Variant::operator PackedInt32Array() const { return _to_packed<int32_t>(*this, PACKED_INT32_ARRAY, _ptr); }
Variant::operator PackedFloat32Array() const { return _to_packed<float>(*this, PACKED_FLOAT32_ARRAY, _ptr); }
Variant::operator PackedVector3Array() const { return _to_packed<Vector3>(*this, PACKED_VECTOR3_ARRAY, _ptr); }

// --- names and strings --------------------------------------------------------

String Variant::get_type_name(Type p_type) {
	static const char *names[VARIANT_MAX] = {
		"Nil", "bool", "int", "float", "String", "Vector2", "Vector2i", "Rect2", "Rect2i", "Vector3",
		"Vector3i", "Transform2D", "Vector4", "Vector4i", "Plane", "Quaternion", "AABB", "Basis", "Transform3D", "Projection",
		"Color", "StringName", "NodePath", "RID", "Object", "Callable", "Signal", "Dictionary", "Array", "PackedByteArray",
		"PackedInt32Array", "PackedInt64Array", "PackedFloat32Array", "PackedFloat64Array", "PackedStringArray", "PackedVector2Array", "PackedVector3Array", "PackedColorArray", "PackedVector4Array"
	};
	return (p_type >= 0 && p_type < VARIANT_MAX) ? String(names[p_type]) : String("<invalid>");
}

static String _v3(const Vector3 &p_v) {
	return "(" + String::num_real(p_v.x, false) + ", " + String::num_real(p_v.y, false) + ", " + String::num_real(p_v.z, false) + ")";
}

template <typename T, typename F>
static String _join(const Vector<T> &p_v, F p_f) {
	String s = "[";
	for (int64_t i = 0; i < p_v.size(); i++) {
		if (i) {
			s += ", ";
		}
		s += p_f(p_v[i]);
	}
	return s + "]";
}

String Variant::stringify() const {
	switch (type) {
		case NIL:
			return "<null>";
		case BOOL:
			return _data._bool ? "true" : "false";
		case INT:
			return itos(_data._int);
		case FLOAT:
			return String::num_real(_data._float, true);
		case STRING:
			return *_payload<String>();
		case VECTOR3:
			return _v3(*_payload<Vector3>());
		case OBJECT: {
			if (!_data._obj) {
				return "<Object#null>";
			}
			char buf[32];
			snprintf(buf, sizeof(buf), "%p", (void *)_data._obj);
			return "<" + _data._obj->get_class() + "#" + String(buf) + ">";
		}
		case CALLABLE:
			return "Callable(" + _payload<Callable>()->get_method() + ")";
		case DICTIONARY: {
			const Dictionary &d = *_payload<Dictionary>();
			String s = "{ ";
			bool first = true;
			for (const Variant *k = d.next(nullptr); k; k = d.next(k)) {
				if (!first) {
					s += ", ";
				}
				first = false;
				s += k->stringify() + ": " + d[*k].stringify();
			}
			return s + " }";
		}
		case ARRAY: {
			const Array &a = *_payload<Array>();
			String s = "[";
			for (int i = 0; i < a.size(); i++) {
				if (i) {
					s += ", ";
				}
				s += a[i].stringify();
			}
			return s + "]";
		}
		case PACKED_INT32_ARRAY:
			return _join(*_payload<PackedInt32Array>(), [](int32_t p_e) { return itos(p_e); });
		case PACKED_FLOAT32_ARRAY:
			return _join(*_payload<PackedFloat32Array>(), [](float p_e) { return String::num_real(p_e, true); });
		case PACKED_VECTOR3_ARRAY:
			return _join(*_payload<PackedVector3Array>(), [](const Vector3 &p_e) { return _v3(p_e); });
		default:
			return "<" + get_type_name(type) + ">";
	}
}

// --- hashing and comparison ---------------------------------------------------

template <typename T>
static uint32_t _hash_packed(const Vector<T> &p_v) {
	uint32_t h = HASH_MURMUR3_SEED;
	for (int64_t i = 0; i < p_v.size(); i++) {
		h = hash_murmur3_one_32(HashMapHasherDefault::hash(p_v[i]), h);
	}
	return hash_fmix32(h);
}

uint32_t Variant::hash() const {
	switch (type) {
		case NIL:
			return 0;
		case BOOL:
			return _data._bool ? 1 : 0;
		case INT:
			return hash_one_uint64(uint64_t(_data._int));
		case FLOAT:
			return hash_murmur3_one_double(_data._float);
		case STRING:
			return _payload<String>()->hash();
		case VECTOR3:
			return _payload<Vector3>()->hash();
		case OBJECT:
			return hash_one_uint64(uint64_t(uintptr_t(_data._obj)));
		case CALLABLE:
			return _payload<Callable>()->hash();
		case DICTIONARY:
			return _payload<Dictionary>()->hash();
		case ARRAY:
			return _payload<Array>()->hash();
		case PACKED_INT32_ARRAY:
			return _hash_packed(*_payload<PackedInt32Array>());
		case PACKED_FLOAT32_ARRAY:
			return _hash_packed(*_payload<PackedFloat32Array>());
		case PACKED_VECTOR3_ARRAY:
			return _hash_packed(*_payload<PackedVector3Array>());
		default:
			return 0;
	}
}

static bool _same_real(double p_a, double p_b) {
	return p_a == p_b || (std::isnan(p_a) && std::isnan(p_b));
}

static bool _same_v3(const Vector3 &p_a, const Vector3 &p_b) {
	return _same_real(p_a.x, p_b.x) && _same_real(p_a.y, p_b.y) && _same_real(p_a.z, p_b.z);
}

template <typename T, typename F>
static bool _same_packed(const Vector<T> &p_a, const Vector<T> &p_b, F p_same) {
	if (p_a.size() != p_b.size()) {
		return false;
	}
	for (int64_t i = 0; i < p_a.size(); i++) {
		if (!p_same(p_a[i], p_b[i])) {
			return false;
		}
	}
	return true;
}

static bool _compare(const Variant &p_a, const Variant &p_b, bool p_strict);

static bool _compare_arrays(const Array &p_a, const Array &p_b, bool p_strict) {
	if (p_a.is_same_instance(p_b)) {
		return true;
	}
	if (p_a.size() != p_b.size()) {
		return false;
	}
	for (int i = 0; i < p_a.size(); i++) {
		if (!_compare(p_a[i], p_b[i], p_strict)) {
			return false;
		}
	}
	return true;
}

static bool _compare_dictionaries(const Dictionary &p_a, const Dictionary &p_b, bool p_strict) {
	if (p_a.is_same_instance(p_b)) {
		return true;
	}
	if (p_a.size() != p_b.size()) {
		return false;
	}
	for (const Variant *k = p_a.next(nullptr); k; k = p_a.next(k)) {
		const Variant *other = p_b.getptr(*k);
		if (!other || !_compare(p_a[*k], *other, p_strict)) {
			return false;
		}
	}
	return true;
}

// p_strict: Dictionary-key semantics (no int/float mixing, NaN == NaN).
static bool _compare(const Variant &p_a, const Variant &p_b, bool p_strict) {
	const Variant::Type ta = p_a.get_type();
	const Variant::Type tb = p_b.get_type();
	if (ta != tb) {
		if (!p_strict && (ta == Variant::INT || ta == Variant::FLOAT) && (tb == Variant::INT || tb == Variant::FLOAT)) {
			return double(p_a) == double(p_b);
		}
		if (!p_strict && p_a.is_null() && p_b.is_null()) {
			return true;
		}
		return false;
	}
	switch (ta) {
		case Variant::NIL:
			return true;
		case Variant::BOOL:
			return bool(p_a) == bool(p_b);
		case Variant::INT:
			return (long long)p_a == (long long)p_b;
		case Variant::FLOAT:
			return p_strict ? _same_real(double(p_a), double(p_b)) : double(p_a) == double(p_b);
		case Variant::STRING:
			return String(p_a) == String(p_b);
		case Variant::VECTOR3:
			return p_strict ? _same_v3(Vector3(p_a), Vector3(p_b)) : Vector3(p_a) == Vector3(p_b);
		case Variant::OBJECT:
			return p_a.get_validated_object() == p_b.get_validated_object();
		case Variant::CALLABLE:
			return Callable(p_a) == Callable(p_b);
		case Variant::DICTIONARY:
			return _compare_dictionaries(Dictionary(p_a), Dictionary(p_b), p_strict);
		case Variant::ARRAY:
			return _compare_arrays(Array(p_a), Array(p_b), p_strict);
		case Variant::PACKED_INT32_ARRAY:
			return PackedInt32Array(p_a) == PackedInt32Array(p_b);
		case Variant::PACKED_FLOAT32_ARRAY:
			return p_strict ? _same_packed(PackedFloat32Array(p_a), PackedFloat32Array(p_b), [](float p_x, float p_y) { return _same_real(p_x, p_y); }) : PackedFloat32Array(p_a) == PackedFloat32Array(p_b);
		case Variant::PACKED_VECTOR3_ARRAY:
			return p_strict ? _same_packed(PackedVector3Array(p_a), PackedVector3Array(p_b), [](const Vector3 &p_x, const Vector3 &p_y) { return _same_v3(p_x, p_y); }) : PackedVector3Array(p_a) == PackedVector3Array(p_b);
		default:
			return false;
	}
}

bool Variant::operator==(const Variant &p_other) const {
	return _compare(*this, p_other, false);
}

bool Variant::hash_compare(const Variant &p_other) const {
	return _compare(*this, p_other, true);
}

} // namespace gdl
