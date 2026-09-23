// godot-lite: Array, a shared reference to a vector of Variants (copying an
// Array aliases it, as in Godot; duplicate() makes a new one).
#pragma once

#include "core/error/error_list.h"
#include "core/typedefs.h"

#include <cstdint>
#include <initializer_list>
#include <memory>

namespace gdl {

class Variant;
struct ArrayPrivate;

class Array {
	std::shared_ptr<ArrayPrivate> _p;

public:
	Array();
	Array(const Array &p_from) = default;
	Array(Array &&p_from);
	Array(std::initializer_list<Variant> p_init);
	Array &operator=(const Array &p_from) = default;
	Array &operator=(Array &&p_from);
	~Array();

	Variant &operator[](int64_t p_idx);
	const Variant &operator[](int64_t p_idx) const;

	int size() const;
	bool is_empty() const;
	void clear();
	Error resize(int64_t p_new_size);

	void push_back(const Variant &p_value);
	void append(const Variant &p_value) { push_back(p_value); }
	Error insert(int64_t p_pos, const Variant &p_value);
	void remove_at(int64_t p_pos);
	void erase(const Variant &p_value);

	int find(const Variant &p_value, int64_t p_from = 0) const;
	bool has(const Variant &p_value) const;

	Array duplicate(bool p_deep = false) const;

	bool operator==(const Array &p_array) const;
	bool operator!=(const Array &p_array) const { return !(*this == p_array); }
	uint32_t hash() const;
	// gdl extension: whether two Arrays are the same shared array.
	bool is_same_instance(const Array &p_other) const { return _p == p_other._p; }
};

} // namespace gdl

#include "core/variant/variant.h"
