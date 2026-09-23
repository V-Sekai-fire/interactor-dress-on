#include "core/variant/array.h"

#include "core/templates/hashfuncs.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <vector>

namespace gdl {

struct ArrayPrivate {
	std::vector<Variant> v;
};

Array::Array() :
		_p(std::make_shared<ArrayPrivate>()) {}

// Godot has no moved-from Array: a "move" aliases like a copy.
Array::Array(Array &&p_from) :
		_p(p_from._p) {}

Array &Array::operator=(Array &&p_from) {
	_p = p_from._p;
	return *this;
}

Array::Array(std::initializer_list<Variant> p_init) :
		_p(std::make_shared<ArrayPrivate>()) {
	_p->v.assign(p_init.begin(), p_init.end());
}

Array::~Array() = default;

Variant &Array::operator[](int64_t p_idx) {
	CRASH_BAD_INDEX(p_idx, int64_t(_p->v.size()));
	return _p->v[size_t(p_idx)];
}

const Variant &Array::operator[](int64_t p_idx) const {
	CRASH_BAD_INDEX(p_idx, int64_t(_p->v.size()));
	return _p->v[size_t(p_idx)];
}

int Array::size() const {
	return int(_p->v.size());
}

bool Array::is_empty() const {
	return _p->v.empty();
}

void Array::clear() {
	_p->v.clear();
}

Error Array::resize(int64_t p_new_size) {
	ERR_FAIL_COND_V(p_new_size < 0, ERR_INVALID_PARAMETER);
	_p->v.resize(size_t(p_new_size));
	return OK;
}

void Array::push_back(const Variant &p_value) {
	_p->v.push_back(p_value);
}

Error Array::insert(int64_t p_pos, const Variant &p_value) {
	if (p_pos < 0) {
		p_pos += size();
	}
	ERR_FAIL_INDEX_V_MSG(p_pos, int64_t(size()) + 1, ERR_INVALID_PARAMETER, "The calculated index is out of bounds (the array has " + itos(size()) + " elements).");
	_p->v.insert(_p->v.begin() + p_pos, p_value);
	return OK;
}

void Array::remove_at(int64_t p_pos) {
	if (p_pos < 0) {
		p_pos += size();
	}
	ERR_FAIL_INDEX(p_pos, int64_t(size()));
	_p->v.erase(_p->v.begin() + p_pos);
}

void Array::erase(const Variant &p_value) {
	const int idx = find(p_value);
	if (idx >= 0) {
		remove_at(idx);
	}
}

int Array::find(const Variant &p_value, int64_t p_from) const {
	if (p_from < 0) {
		p_from += size();
		if (p_from < 0) {
			p_from = 0;
		}
	}
	for (int64_t i = p_from; i < int64_t(_p->v.size()); i++) {
		if (_p->v[size_t(i)] == p_value) {
			return int(i);
		}
	}
	return -1;
}

bool Array::has(const Variant &p_value) const {
	return find(p_value) != -1;
}

static Variant _duplicate_variant(const Variant &p_v, bool p_deep) {
	if (!p_deep) {
		return p_v;
	}
	if (p_v.get_type() == Variant::ARRAY) {
		return Array(p_v).duplicate(true);
	}
	if (p_v.get_type() == Variant::DICTIONARY) {
		return Dictionary(p_v).duplicate(true);
	}
	return p_v; // Packed arrays are values; Objects are shared, as in Godot.
}

Array Array::duplicate(bool p_deep) const {
	Array r;
	r._p->v.reserve(_p->v.size());
	for (const Variant &e : _p->v) {
		r._p->v.push_back(_duplicate_variant(e, p_deep));
	}
	return r;
}

bool Array::operator==(const Array &p_array) const {
	return Variant(*this) == Variant(p_array);
}

uint32_t Array::hash() const {
	uint32_t h = hash_murmur3_one_32(uint32_t(Variant::ARRAY));
	for (const Variant &e : _p->v) {
		h = hash_murmur3_one_32(e.hash(), h);
	}
	return hash_fmix32(h);
}

} // namespace gdl
