// godot-lite: TypedArray<T> is an Array (same shared storage); the element
// type is documentation only, not checked.
#pragma once

#include "core/variant/array.h"
#include "core/variant/variant.h"

#include <initializer_list>

namespace gdl {

template <typename T>
class TypedArray : public Array {
public:
	TypedArray() = default;
	TypedArray(const Array &p_array) :
			Array(p_array) {}
	TypedArray(const Variant &p_variant) :
			Array(p_variant.operator Array()) {}
	TypedArray(std::initializer_list<Variant> p_init) :
			Array(p_init) {}
	TypedArray &operator=(const Array &p_array) {
		Array::operator=(p_array);
		return *this;
	}
};

} // namespace gdl
