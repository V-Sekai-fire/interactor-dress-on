// godot-lite: Variant with exactly the types Cassie moves through
// Dictionaries, Arrays and Callables: nil, bool, int, float, String, Vector3,
// Object (holding a reference when RefCounted), Callable, Dictionary, Array,
// PackedInt32Array, PackedFloat32Array and PackedVector3Array. Any other type
// has no constructor, so putting one in a Variant fails to compile rather
// than misbehaving. The type enum keeps Godot's full numbering. Non-scalar
// payloads are immutable and shared between copies.
#pragma once

#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/plane.h"
#include "core/math/quaternion.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

#include <memory>

namespace gdl {

class Object;
class Array;
class Dictionary;
class Callable;

typedef Vector<uint8_t> PackedByteArray;
typedef Vector<int32_t> PackedInt32Array;
typedef Vector<int64_t> PackedInt64Array;
typedef Vector<float> PackedFloat32Array;
typedef Vector<double> PackedFloat64Array;
typedef Vector<real_t> PackedRealArray;
typedef Vector<String> PackedStringArray;
typedef Vector<Vector2> PackedVector2Array;
typedef Vector<Vector3> PackedVector3Array;

class Variant {
public:
	enum Type {
		NIL,
		BOOL,
		INT,
		FLOAT,
		STRING,
		VECTOR2,
		VECTOR2I,
		RECT2,
		RECT2I,
		VECTOR3,
		VECTOR3I,
		TRANSFORM2D,
		VECTOR4,
		VECTOR4I,
		PLANE,
		QUATERNION,
		AABB,
		BASIS,
		TRANSFORM3D,
		PROJECTION,
		COLOR,
		STRING_NAME,
		NODE_PATH,
		RID,
		OBJECT,
		CALLABLE,
		SIGNAL,
		DICTIONARY,
		ARRAY,
		PACKED_BYTE_ARRAY,
		PACKED_INT32_ARRAY,
		PACKED_INT64_ARRAY,
		PACKED_FLOAT32_ARRAY,
		PACKED_FLOAT64_ARRAY,
		PACKED_STRING_ARRAY,
		PACKED_VECTOR2_ARRAY,
		PACKED_VECTOR3_ARRAY,
		PACKED_COLOR_ARRAY,
		PACKED_VECTOR4_ARRAY,
		VARIANT_MAX
	};

private:
	Type type = NIL;
	union {
		bool _bool;
		int64_t _int;
		double _float;
		Object *_obj;
	} _data = {};
	// Payload of every non-scalar type; for OBJECT, owns one reference when
	// the object is RefCounted.
	std::shared_ptr<const void> _ptr;

	template <typename T>
	void _set_payload(Type p_type, const T &p_value) {
		type = p_type;
		_ptr = std::make_shared<const T>(p_value);
	}
	template <typename T>
	const T *_payload() const { return static_cast<const T *>(_ptr.get()); }

public:
	Variant() = default;
	Variant(const Variant &) = default;
	Variant(Variant &&) = default;
	Variant &operator=(const Variant &) = default;
	Variant &operator=(Variant &&) = default;
	~Variant() = default;

	Variant(bool p_bool);
	Variant(signed char p_int);
	Variant(unsigned char p_int);
	Variant(short p_int);
	Variant(unsigned short p_int);
	Variant(int p_int);
	Variant(unsigned int p_int);
	Variant(long p_int);
	Variant(unsigned long p_int);
	Variant(long long p_int);
	Variant(unsigned long long p_int);
	Variant(float p_float);
	Variant(double p_float);
	Variant(const String &p_string);
	Variant(const char *p_string);
	Variant(const Vector3 &p_v);
	Variant(const Object *p_object);
	Variant(const Callable &p_callable);
	Variant(const Dictionary &p_dictionary);
	Variant(const Array &p_array);
	Variant(const PackedInt32Array &p_v);
	Variant(const PackedFloat32Array &p_v);
	Variant(const PackedVector3Array &p_v);

	Type get_type() const { return type; }
	bool is_null() const { return type == NIL || (type == OBJECT && _data._obj == nullptr); }
	static String get_type_name(Type p_type);

	operator bool() const;
	// Every standard integer type, so int64_t/size_t resolve on LP64 and LLP64.
	operator long long() const;
	operator signed char() const { return (signed char)operator long long(); }
	operator unsigned char() const { return (unsigned char)operator long long(); }
	operator short() const { return short(operator long long()); }
	operator unsigned short() const { return (unsigned short)operator long long(); }
	operator int() const { return int(operator long long()); }
	operator unsigned int() const { return (unsigned int)operator long long(); }
	operator long() const { return long(operator long long()); }
	operator unsigned long() const { return (unsigned long)operator long long(); }
	operator unsigned long long() const { return (unsigned long long)operator long long(); }
	operator float() const { return float(operator double()); }
	operator double() const;
	operator String() const;
	operator Vector3() const;
	operator Object *() const { return get_validated_object(); }
	operator Callable() const;
	operator Dictionary() const;
	operator Array() const;
	operator PackedInt32Array() const;
	operator PackedFloat32Array() const;
	operator PackedVector3Array() const;

	Object *get_validated_object() const { return type == OBJECT ? _data._obj : nullptr; }

	bool booleanize() const { return operator bool(); }
	String stringify() const;
	uint32_t hash() const;
	// Godot's `==` between Variants: same type (or int/float) and equal value;
	// Objects, Arrays, Dictionaries and Callables compare by identity/contents
	// as in Godot (containers by value, recursively).
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const { return !(*this == p_other); }
	// Strict hashing comparison (int 1 != float 1.0) used by Dictionary keys.
	bool hash_compare(const Variant &p_other) const;
};

// Dictionary keys: Variant::hash + Variant::hash_compare.
struct VariantHasher {
	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};
struct VariantComparator {
	static _FORCE_INLINE_ bool compare(const Variant &p_lhs, const Variant &p_rhs) { return p_lhs.hash_compare(p_rhs); }
};

String vformat_array(const String &p_text, const Array &p_args);

template <typename... VarArgs>
String vformat(const String &p_text, const VarArgs... p_args);

} // namespace gdl

// array.h includes variant.h at its end, so either include order leaves
// both complete before vformat below. Godot's variant.h also brings in
// Dictionary and Callable; Cassie relies on that.
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"

namespace gdl {

template <typename... VarArgs>
String vformat(const String &p_text, const VarArgs... p_args) {
	Array args;
	(args.push_back(Variant(p_args)), ...);
	return vformat_array(p_text, args);
}

} // namespace gdl
