// A stackful fiber for the riscv64 guest: one heap stack, one hand-written
// context switch (fiber_riscv64.S), nothing from the libc.
//
// Why: a guest job that submits GPU work must not sync() in the same frame
// (AGENTS.md rule 4). With a fiber the job is written as straight-line code
// that yields WAIT_GPU after each submit; the host's next vmcall resumes it
// and it syncs then. The fiber's stack lives on the guest heap, so it
// survives between vmcalls; the caller's side is saved afresh on every
// resume(), so each vmcall's own stack is only borrowed for the duration of
// that call. Gate 0F probe 11 is the evidence that this holds in
// godot-sandbox (100 submit/yield/resume rounds, throw/catch inside).
//
// Not thread-safe and not reentrant: one resume() at a time, from the guest's
// only hart.
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {
// Callee-saved state of one side of a switch, per the RISC-V LP64D psABI:
// ra, sp, s0-s11 and fs0-fs11. Offsets are fixed by fiber_riscv64.S.
struct fiber_ctx {
	uint64_t ra;
	uint64_t sp;
	uint64_t s[12];
	uint64_t fs[12];
};
static_assert(sizeof(fiber_ctx) == 208, "fiber_riscv64.S hard-codes this layout");

// Save the current callee-saved registers into *from, load *to, return into
// to->ra with to->sp.
void fiber_switch(fiber_ctx *from, const fiber_ctx *to);
// First frame of a new fiber: calls s1(s0) and never returns.
void fiber_trampoline();
}

class Fiber;
// The C entry the trampoline calls (fiber.cpp).
extern "C" void fiber_main_c(Fiber *self);

class Fiber {
public:
	using Entry = void (*)(Fiber &self, void *arg);

	Fiber(Entry entry, void *arg, size_t stack_bytes = 256 * 1024);
	~Fiber();
	Fiber(const Fiber &) = delete;
	Fiber &operator=(const Fiber &) = delete;

	// Run the fiber until it yields or returns. Returns true while it has
	// more to do (it yielded), false once the body has returned.
	bool resume();
	// Inside the fiber: switch back to the resume() that ran us.
	void yield();

	bool done() const { return done_; }
	// An exception escaped the body (it is swallowed at the fiber's base,
	// since nothing above the trampoline can unwind into the caller's stack).
	bool escaped() const { return escaped_; }
	size_t stack_bytes() const { return stack_bytes_; }

	// Free for the body to say why it yielded (e.g. WAIT_GPU).
	int status = 0;

private:
	friend void fiber_main_c(Fiber *self);

	fiber_ctx self_{};
	fiber_ctx caller_{};
	Entry entry_;
	void *arg_;
	uint8_t *stack_ = nullptr;
	size_t stack_bytes_;
	bool started_ = false;
	bool done_ = false;
	bool escaped_ = false;
};
