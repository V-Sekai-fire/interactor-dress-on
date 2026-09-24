// The kernel runners gen_host_kernels.py writes: kernel `id` (its line in
// kernels/ggml/kernels.txt) over `groups`, its 64 params words at set 1 base 0,
// and one memory block per storage binding (s0, s1, s2, dst). Every element
// offset in the words is relative to its binding's block, as on the GPU where
// each binding is its own RD buffer. The host harness binds all four to one
// block; the guest's CPU fallback binds each to its ggml buffer's memory.
// False if the kernel has no cpp emit and no sibling that runs in its place.
#pragma once
#include <cstddef>
#include <cstdint>

struct KernelBinding {
	void *mem = nullptr;
	size_t bytes = 0;
};

bool run_kernel(int id, uint32_t *words, const KernelBinding bindings[4], const uint32_t groups[3]);
bool run_kernel(int id, uint32_t *words, void *mem, size_t bytes, const uint32_t groups[3]);
