// godot-lite: Godot's copy-on-write Vector<T> over a shared std::vector.
// Copies share storage; the first write through `.write[]`, ptrw(), set(),
// resize() or any other mutator detaches (clones) when the storage is shared.
// Vector<bool> stores bytes so ptr()/ptrw() stay real pointers.
#pragma once

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/templates/sort_array.h"
#include "core/templates/span.h"
#include "core/typedefs.h"

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

namespace gdl {

template <typename T>
class Vector;

template <typename T>
class VectorWriteProxy {
public:
	_FORCE_INLINE_ T &operator[](int64_t p_index) {
		// `write` is Vector's first member, so its address is the Vector's.
		Vector<T> *v = reinterpret_cast<Vector<T> *>(this);
		CRASH_BAD_INDEX(p_index, v->size());
		return v->ptrw()[p_index];
	}
};

template <typename T>
class Vector {
	friend class VectorWriteProxy<T>;
	using Store = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
	static_assert(sizeof(Store) == sizeof(T), "Vector storage must alias T");

public:
	VectorWriteProxy<T> write;
	typedef T ValueType;
	using Size = int64_t;
	using USize = uint64_t;

private:
	std::shared_ptr<std::vector<Store>> _data;

	const std::vector<Store> *_read() const { return _data.get(); }
	std::vector<Store> &_mut() {
		if (!_data) {
			_data = std::make_shared<std::vector<Store>>();
		} else if (_data.use_count() > 1) {
			_data = std::make_shared<std::vector<Store>>(*_data);
		}
		return *_data;
	}

public:
	// gdl extension: whether two Vectors share one buffer (for COW tests).
	bool shares_storage_with(const Vector &p_other) const { return _data && _data == p_other._data; }

	_FORCE_INLINE_ bool push_back(T p_elem) {
		_mut().push_back(Store(std::move(p_elem)));
		return false;
	}
	_FORCE_INLINE_ bool append(T p_elem) { return push_back(std::move(p_elem)); }
	void fill(T p_elem) {
		T *p = ptrw();
		for (Size i = 0; i < size(); i++) {
			p[i] = p_elem;
		}
	}

	void remove_at(Size p_index) {
		ERR_FAIL_INDEX(p_index, size());
		std::vector<Store> &v = _mut();
		v.erase(v.begin() + p_index);
	}
	_FORCE_INLINE_ bool erase(const T &p_val) {
		Size idx = find(p_val);
		if (idx >= 0) {
			remove_at(idx);
			return true;
		}
		return false;
	}

	void reverse() {
		T *p = ptrw();
		for (Size i = 0; i < size() / 2; i++) {
			SWAP(p[i], p[size() - i - 1]);
		}
	}

	_FORCE_INLINE_ T *ptrw() {
		if (size() == 0) {
			return nullptr;
		}
		return reinterpret_cast<T *>(_mut().data());
	}
	_FORCE_INLINE_ const T *ptr() const {
		const std::vector<Store> *v = _read();
		return (v && !v->empty()) ? reinterpret_cast<const T *>(v->data()) : nullptr;
	}
	_FORCE_INLINE_ Size size() const {
		const std::vector<Store> *v = _read();
		return v ? Size(v->size()) : 0;
	}
	_FORCE_INLINE_ USize capacity() const {
		const std::vector<Store> *v = _read();
		return v ? USize(v->capacity()) : 0;
	}

	_FORCE_INLINE_ operator Span<T>() const { return span(); }
	_FORCE_INLINE_ Span<T> span() const { return Span<T>(ptr(), uint64_t(size())); }

	_FORCE_INLINE_ void clear() { _data.reset(); }
	_FORCE_INLINE_ bool is_empty() const { return size() == 0; }

	_FORCE_INLINE_ T get(Size p_index) {
		CRASH_BAD_INDEX(p_index, size());
		return ptr()[p_index];
	}
	_FORCE_INLINE_ const T &get(Size p_index) const {
		CRASH_BAD_INDEX(p_index, size());
		return ptr()[p_index];
	}
	_FORCE_INLINE_ void set(Size p_index, const T &p_elem) {
		ERR_FAIL_INDEX(p_index, size());
		ptrw()[p_index] = p_elem;
	}

	// New elements are value-initialized (zeroed for trivial types), where
	// Godot leaves trivial types uninitialized: a superset of its contract.
	_FORCE_INLINE_ Error resize(Size p_size) {
		ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);
		if (p_size == size()) {
			return OK;
		}
		if (p_size == 0) {
			clear();
			return OK;
		}
		_mut().resize(size_t(p_size));
		return OK;
	}
	_FORCE_INLINE_ Error resize_initialized(Size p_size) { return resize(p_size); }
	_FORCE_INLINE_ Error resize_uninitialized(Size p_size) { return resize(p_size); }

	Error reserve(Size p_size) {
		ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);
		_mut().reserve(size_t(p_size));
		return OK;
	}
	Error reserve_exact(Size p_size) { return reserve(p_size); }

	_FORCE_INLINE_ const T &operator[](Size p_index) const {
		CRASH_BAD_INDEX(p_index, size());
		return ptr()[p_index];
	}

	Error insert(Size p_pos, T p_val) {
		ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
		std::vector<Store> &v = _mut();
		v.insert(v.begin() + p_pos, Store(std::move(p_val)));
		return OK;
	}
	Size find(const T &p_val, Size p_from = 0) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_from < 0 || p_from >= size()) {
			return -1;
		}
		const T *p = ptr();
		for (Size i = p_from; i < size(); i++) {
			if (p[i] == p_val) {
				return i;
			}
		}
		return -1;
	}
	Size rfind(const T &p_val, Size p_from = -1) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_from < 0 || p_from >= size()) {
			return -1;
		}
		const T *p = ptr();
		for (Size i = p_from; i >= 0; i--) {
			if (p[i] == p_val) {
				return i;
			}
		}
		return -1;
	}
	Size count(const T &p_val) const {
		Size c = 0;
		const T *p = ptr();
		for (Size i = 0; i < size(); i++) {
			c += (p[i] == p_val) ? 1 : 0;
		}
		return c;
	}

	void append_array(const Vector<T> &p_other) {
		if (p_other.is_empty()) {
			return;
		}
		// Copy first: p_other may share (or be) this storage.
		const std::vector<Store> other = *p_other._data;
		std::vector<Store> &v = _mut();
		v.insert(v.end(), other.begin(), other.end());
	}
	void append_array(Span<T> p_other) {
		for (uint64_t i = 0; i < p_other.size(); i++) {
			push_back(p_other.ptr()[i]);
		}
	}

	_FORCE_INLINE_ bool has(const T &p_val) const { return find(p_val) != -1; }

	void sort() {
		sort_custom<Comparator<T>>();
	}

	template <typename C, bool Validate = SORT_ARRAY_VALIDATE_ENABLED, typename... Args>
	void sort_custom(Args &&...p_args) {
		Size len = size();
		if (len == 0) {
			return;
		}
		T *data = ptrw();
		SortArray<T, C, Validate> sorter{ p_args... };
		sorter.sort(data, len);
	}

	Size bsearch(const T &p_value, bool p_before) const {
		return bsearch_custom<Comparator<T>>(p_value, p_before);
	}

	template <typename C, typename Value, typename... Args>
	Size bsearch_custom(const Value &p_value, bool p_before, Args &&...p_args) const {
		return span().bisect(p_value, p_before, C{ p_args... });
	}

	Vector<T> duplicate() const {
		return *this;
	}

	void ordered_insert(const T &p_val) {
		Size idx = span().bisect(p_val, false);
		insert(idx, p_val);
	}

	void operator=(const Vector &p_from) { _data = p_from._data; }
	void operator=(Vector &&p_from) { _data = std::move(p_from._data); }

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		if (is_empty()) {
			return ret;
		}
		size_t alloc_size = size_t(size()) * sizeof(T);
		ret.resize(Size(alloc_size));
		memcpy(ret.ptrw(), ptr(), alloc_size);
		return ret;
	}

	Vector<T> slice(Size p_begin, Size p_end = INT64_MAX) const {
		Vector<T> result;
		const Size s = size();
		Size begin = CLAMP(p_begin, -s, s);
		if (begin < 0) {
			begin += s;
		}
		Size end = CLAMP(p_end, -s, s);
		if (end < 0) {
			end += s;
		}
		ERR_FAIL_COND_V(begin > end, result);
		const Size result_size = end - begin;
		result.resize(result_size);
		const T *const r = ptr();
		T *const w = result.ptrw();
		for (Size i = 0; i < result_size; ++i) {
			w[i] = r[begin + i];
		}
		return result;
	}

	bool operator==(const Vector<T> &p_arr) const {
		if (size() != p_arr.size()) {
			return false;
		}
		const T *a = ptr();
		const T *b = p_arr.ptr();
		for (Size i = 0; i < size(); i++) {
			if (!(a[i] == b[i])) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const Vector<T> &p_arr) const { return !(*this == p_arr); }

	// Iteration. begin()/end() on a non-const Vector detach, like Godot's.
	T *begin() { return ptrw(); }
	T *end() { return ptrw() + size(); }
	const T *begin() const { return ptr(); }
	const T *end() const { return ptr() + size(); }

	_FORCE_INLINE_ Vector() {}
	_FORCE_INLINE_ Vector(std::initializer_list<T> p_init) {
		if (p_init.size()) {
			_data = std::make_shared<std::vector<Store>>();
			_data->reserve(p_init.size());
			for (const T &e : p_init) {
				_data->push_back(Store(e));
			}
		}
	}
	_FORCE_INLINE_ explicit Vector(Span<T> p_span) { append_array(p_span); }
	_FORCE_INLINE_ Vector(const Vector &p_from) :
			_data(p_from._data) {}
	_FORCE_INLINE_ Vector(Vector &&p_from) :
			_data(std::move(p_from._data)) {}
};

} // namespace gdl
