// godot-lite: Dictionary, a shared reference to an insertion-ordered
// Variant -> Variant map (Godot's own HashMap from the vendored core).
#pragma once

#include "core/variant/array.h"
#include "core/variant/variant.h"

#include <memory>

namespace gdl {

struct DictionaryPrivate;

class Dictionary {
	std::shared_ptr<DictionaryPrivate> _p;

public:
	Dictionary();
	Dictionary(const Dictionary &p_from) = default;
	Dictionary(Dictionary &&p_from);
	Dictionary &operator=(const Dictionary &p_from) = default;
	Dictionary &operator=(Dictionary &&p_from);
	~Dictionary();

	// Inserts nil on a missing key (Godot's non-const operator[]).
	Variant &operator[](const Variant &p_key);
	// Prints an error and returns nil on a missing key.
	const Variant &operator[](const Variant &p_key) const;

	Variant get(const Variant &p_key, const Variant &p_default) const;
	const Variant *getptr(const Variant &p_key) const;

	int size() const;
	bool is_empty() const;
	void clear();
	bool has(const Variant &p_key) const;
	bool erase(const Variant &p_key);

	// Key iteration in insertion order: next(nullptr) is the first key,
	// next(key) the one after it, nullptr past the end.
	const Variant *next(const Variant *p_key = nullptr) const;
	Array keys() const;
	Array values() const;
	Dictionary duplicate(bool p_deep = false) const;

	bool operator==(const Dictionary &p_dictionary) const;
	bool operator!=(const Dictionary &p_dictionary) const { return !(*this == p_dictionary); }
	uint32_t hash() const;
	// gdl extension: whether two Dictionaries are the same shared map.
	bool is_same_instance(const Dictionary &p_other) const { return _p == p_other._p; }
};

} // namespace gdl
