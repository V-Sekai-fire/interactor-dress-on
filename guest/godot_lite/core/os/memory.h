// godot-lite: the part of Godot's allocation API that Cassie and the vendored
// templates use (memnew/memdelete, Memory::*_static, the allocator and
// placement helpers) over malloc/realloc/free. No usage accounting;
// `p_pad_align` keeps Godot's size-header layout.
#pragma once

#include "core/error/error_macros.h"
#include "core/typedefs.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>

namespace gdl {

namespace Memory {
constexpr size_t get_aligned_address(size_t p_address, size_t p_alignment) {
	const size_t n_bytes_unaligned = p_address % p_alignment;
	return (n_bytes_unaligned == 0) ? p_address : (p_address + p_alignment - n_bytes_unaligned);
}

static constexpr size_t MAX_ALIGN = alignof(max_align_t) < 16 ? 16 : alignof(max_align_t);
inline constexpr size_t SIZE_OFFSET = 0;
inline constexpr size_t ELEMENT_OFFSET = get_aligned_address(SIZE_OFFSET + sizeof(uint64_t), alignof(uint64_t));
inline constexpr size_t DATA_OFFSET = get_aligned_address(ELEMENT_OFFSET + sizeof(uint64_t), MAX_ALIGN);

void *alloc_static_impl(size_t p_bytes, bool p_pad_align, bool p_zeroed);

template <bool p_ensure_zero = false>
inline void *alloc_static(size_t p_bytes, bool p_pad_align = false) {
	return alloc_static_impl(p_bytes, p_pad_align, p_ensure_zero);
}
inline void *alloc_static_zeroed(size_t p_bytes, bool p_pad_align = false) {
	return alloc_static_impl(p_bytes, p_pad_align, true);
}
void *realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align = false);
void free_static(void *p_ptr, bool p_pad_align = false);
} // namespace Memory

class DefaultAllocator {
public:
	static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory, false); }
	static void free(void *p_ptr) { Memory::free_static(p_ptr, false); }
};

} // namespace gdl

// Placement-new tags; they must live at global scope.
void *operator new(size_t p_size, gdl::DefaultAllocator p_allocator);
void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size));
void operator delete(void *p_mem, gdl::DefaultAllocator p_allocator);
void operator delete(void *p_mem, void *(*p_allocfunc)(size_t p_size));

#define memalloc(m_size) ::gdl::Memory::alloc_static(m_size)
#define memrealloc(m_mem, m_size) ::gdl::Memory::realloc_static(m_mem, m_size)
#define memfree(m_mem) ::gdl::Memory::free_static(m_mem)

namespace gdl {

template <typename T>
_ALWAYS_INLINE_ T *_post_initialize(T *p_obj) {
	return p_obj;
}

} // namespace gdl

#define memnew(m_class) ::gdl::_post_initialize(::new (::gdl::DefaultAllocator{}) m_class)
#define memnew_allocator(m_class, m_allocator) ::gdl::_post_initialize(::new (m_allocator::alloc) m_class)
#define memnew_placement(m_placement, m_class) ::gdl::_post_initialize(::new (m_placement) m_class)

namespace gdl {

template <typename T>
void memdelete(T *p_class) {
	if (unlikely(p_class == nullptr)) {
		return;
	}
	if constexpr (!std::is_trivially_destructible_v<T>) {
		p_class->~T();
	}
	Memory::free_static(p_class, false);
}

template <typename T, typename A>
void memdelete_allocator(T *p_class) {
	if (unlikely(p_class == nullptr)) {
		return;
	}
	if constexpr (!std::is_trivially_destructible_v<T>) {
		p_class->~T();
	}
	A::free(p_class);
}

template <typename T>
_FORCE_INLINE_ void memnew_arr_placement(T *p_start, size_t p_num) {
	if constexpr (is_zero_constructible_v<T>) {
		memset(static_cast<void *>(p_start), 0, p_num * sizeof(T));
	} else {
		for (size_t i = 0; i < p_num; i++) {
			memnew_placement(p_start + i, T());
		}
	}
}

// RBMap's sentinel (core/templates/rb_map.h).
struct _GlobalNil {
	int color = 1;
	_GlobalNil *right = nullptr;
	_GlobalNil *left = nullptr;
	_GlobalNil *parent = nullptr;
	_GlobalNil();
};

struct _GlobalNilClass {
	static _GlobalNil _nil;
};

template <typename T>
class DefaultTypedAllocator {
public:
	template <typename... Args>
	_FORCE_INLINE_ T *new_allocation(const Args &&...p_args) { return memnew(T(p_args...)); }
	_FORCE_INLINE_ void delete_allocation(T *p_allocation) { memdelete(p_allocation); }
};

} // namespace gdl
