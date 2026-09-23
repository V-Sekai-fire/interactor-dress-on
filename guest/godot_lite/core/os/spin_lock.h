// godot-lite: SpinLock for PagedAllocator (core/templates/paged_allocator.h).
#pragma once

#include <atomic>

namespace gdl {

class SpinLock {
	mutable std::atomic_flag locked = ATOMIC_FLAG_INIT;

public:
	void lock() const {
		while (locked.test_and_set(std::memory_order_acquire)) {
		}
	}
	void unlock() const {
		locked.clear(std::memory_order_release);
	}
};

} // namespace gdl
