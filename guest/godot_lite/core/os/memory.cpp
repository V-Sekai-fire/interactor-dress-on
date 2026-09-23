#include "core/os/memory.h"

#include <cstdlib>

namespace gdl {

void *Memory::alloc_static_impl(size_t p_bytes, bool p_pad_align, bool p_zeroed) {
	const size_t total = p_bytes + (p_pad_align ? DATA_OFFSET : 0);
	uint8_t *mem = (uint8_t *)(p_zeroed ? calloc(1, total ? total : 1) : malloc(total ? total : 1));
	ERR_FAIL_NULL_V(mem, nullptr);
	if (p_pad_align) {
		*(uint64_t *)(mem + SIZE_OFFSET) = p_bytes;
		return mem + DATA_OFFSET;
	}
	return mem;
}

void *Memory::realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align) {
	if (p_memory == nullptr) {
		return alloc_static(p_bytes, p_pad_align);
	}
	if (p_bytes == 0) {
		free_static(p_memory, p_pad_align);
		return nullptr;
	}
	uint8_t *mem = (uint8_t *)p_memory;
	if (p_pad_align) {
		mem -= DATA_OFFSET;
		mem = (uint8_t *)realloc(mem, p_bytes + DATA_OFFSET);
		ERR_FAIL_NULL_V(mem, nullptr);
		*(uint64_t *)(mem + SIZE_OFFSET) = p_bytes;
		return mem + DATA_OFFSET;
	}
	mem = (uint8_t *)realloc(mem, p_bytes);
	ERR_FAIL_NULL_V(mem, nullptr);
	return mem;
}

void Memory::free_static(void *p_ptr, bool p_pad_align) {
	if (p_ptr == nullptr) {
		return;
	}
	uint8_t *mem = (uint8_t *)p_ptr;
	if (p_pad_align) {
		mem -= DATA_OFFSET;
	}
	free(mem);
}

_GlobalNil::_GlobalNil() {
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;

} // namespace gdl

void *operator new(size_t p_size, gdl::DefaultAllocator) {
	return gdl::Memory::alloc_static(p_size, false);
}

void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)) {
	return p_allocfunc(p_size);
}

void operator delete(void *p_mem, gdl::DefaultAllocator) {
	gdl::Memory::free_static(p_mem, false);
}

void operator delete(void *, void *(*)(size_t)) {
}
