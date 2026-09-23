#include "fiber.h"

#include <cstdlib>

// Called by fiber_trampoline with a0 = the Fiber. Never returns: when the
// body is done it switches back to the last resume() for good.
void fiber_main_c(Fiber *self) {
	try {
		self->entry_(*self, self->arg_);
	} catch (...) {
		self->escaped_ = true;
	}
	self->done_ = true;
	fiber_switch(&self->self_, &self->caller_);
	__builtin_trap(); // a finished fiber is never resumed
}

Fiber::Fiber(Entry entry, void *arg, size_t stack_bytes) :
		entry_(entry), arg_(arg), stack_bytes_(stack_bytes) {
	stack_ = static_cast<uint8_t *>(std::malloc(stack_bytes_));
	if (!stack_) {
		done_ = true;
		return;
	}
	// sp 16-byte aligned at the top; the trampoline finds its arguments in
	// s0 (the Fiber) and s1 (the C entry).
	uintptr_t top = (reinterpret_cast<uintptr_t>(stack_) + stack_bytes_) & ~uintptr_t(15);
	self_.sp = top;
	self_.ra = reinterpret_cast<uint64_t>(&fiber_trampoline);
	self_.s[0] = reinterpret_cast<uint64_t>(this);
	self_.s[1] = reinterpret_cast<uint64_t>(&fiber_main_c);
}

Fiber::~Fiber() {
	std::free(stack_);
}

bool Fiber::resume() {
	if (done_) {
		return false;
	}
	started_ = true;
	fiber_switch(&caller_, &self_);
	return !done_;
}

void Fiber::yield() {
	fiber_switch(&self_, &caller_);
}
