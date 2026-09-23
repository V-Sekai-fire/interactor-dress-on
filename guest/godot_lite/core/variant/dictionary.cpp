#include "core/variant/dictionary.h"

#include "core/templates/hash_map.h"
#include "core/templates/hashfuncs.h"

namespace gdl {

struct DictionaryPrivate {
	HashMap<Variant, Variant, VariantHasher, VariantComparator> map;
};

Dictionary::Dictionary() :
		_p(std::make_shared<DictionaryPrivate>()) {}

// Godot has no moved-from Dictionary: a "move" aliases like a copy.
Dictionary::Dictionary(Dictionary &&p_from) :
		_p(p_from._p) {}

Dictionary &Dictionary::operator=(Dictionary &&p_from) {
	_p = p_from._p;
	return *this;
}

Dictionary::~Dictionary() = default;

Variant &Dictionary::operator[](const Variant &p_key) {
	Variant *v = _p->map.getptr(p_key);
	if (v) {
		return *v;
	}
	return _p->map.insert(p_key, Variant())->value;
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	static const Variant empty;
	const Variant *v = _p->map.getptr(p_key);
	ERR_FAIL_NULL_V_MSG(v, empty, "Bug: Dictionary::operator[] used when there was no value for the given key, please report.");
	return *v;
}

Variant Dictionary::get(const Variant &p_key, const Variant &p_default) const {
	const Variant *v = _p->map.getptr(p_key);
	return v ? *v : p_default;
}

const Variant *Dictionary::getptr(const Variant &p_key) const {
	return _p->map.getptr(p_key);
}

int Dictionary::size() const {
	return int(_p->map.size());
}

bool Dictionary::is_empty() const {
	return _p->map.is_empty();
}

void Dictionary::clear() {
	_p->map.clear();
}

bool Dictionary::has(const Variant &p_key) const {
	return _p->map.has(p_key);
}

bool Dictionary::erase(const Variant &p_key) {
	return _p->map.erase(p_key);
}

const Variant *Dictionary::next(const Variant *p_key) const {
	if (p_key == nullptr) {
		if (_p->map.begin()) {
			return &_p->map.begin()->key;
		}
		return nullptr;
	}
	HashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstIterator E = _p->map.find(*p_key);
	if (!E) {
		return nullptr;
	}
	++E;
	if (E) {
		return &E->key;
	}
	return nullptr;
}

Array Dictionary::keys() const {
	Array a;
	a.resize(size());
	int i = 0;
	for (const KeyValue<Variant, Variant> &kv : _p->map) {
		a[i++] = kv.key;
	}
	return a;
}

Array Dictionary::values() const {
	Array a;
	a.resize(size());
	int i = 0;
	for (const KeyValue<Variant, Variant> &kv : _p->map) {
		a[i++] = kv.value;
	}
	return a;
}

Dictionary Dictionary::duplicate(bool p_deep) const {
	Dictionary d;
	for (const KeyValue<Variant, Variant> &kv : _p->map) {
		Variant v = kv.value;
		if (p_deep && v.get_type() == Variant::ARRAY) {
			v = Array(v).duplicate(true);
		} else if (p_deep && v.get_type() == Variant::DICTIONARY) {
			v = Dictionary(v).duplicate(true);
		}
		d._p->map.insert(kv.key, v);
	}
	return d;
}

bool Dictionary::operator==(const Dictionary &p_dictionary) const {
	return Variant(*this) == Variant(p_dictionary);
}

uint32_t Dictionary::hash() const {
	uint32_t h = hash_murmur3_one_32(uint32_t(Variant::DICTIONARY));
	for (const KeyValue<Variant, Variant> &kv : _p->map) {
		h = hash_murmur3_one_32(kv.key.hash(), h);
		h = hash_murmur3_one_32(kv.value.hash(), h);
	}
	return hash_fmix32(h);
}

} // namespace gdl
