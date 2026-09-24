// ggml-rd's CPU fallback: a dispatch through its kernel's slangc cpp emit.
//
// The same Lean kernel emits both targets (AGENTS.md rule 2): SPIR-V for the
// GPU (rd_kernels.cpp) and cpp for here. The runner is the one the host
// harness uses (tests/ggml_rd_kernels/gen_host_kernels.py writes it into the
// build directory from kernels/ggml/kernels.txt): every emit in a namespace
// of its own, run_kernel(id, words, bindings, groups) walking the grid one
// thread after another. What differs from the harness: each storage binding
// is its own ggml buffer's memory, as on the GPU where each is its own RD
// buffer, so the element offsets fill_standard wrote hold unchanged.
//
// A kernel that shares group memory has no cpp emit; its packer picks the
// serial sibling under serial_kernels(), which graph_compute sets here, and
// the runner maps the rest (the row kernels' `<k>_serial`, mul_mat's
// cpp_siblings.txt pairs) itself.
#include "rd_internal.h"
#include "run_kernel.h"

#include <string>

namespace ggml_rd {

bool cpu_run(const CpuDispatch &d, std::string *why) {
	KernelBinding b[4];
	for (int i = 0; i < 4; ++i) {
		if (d.bind[i] == nullptr || d.bind[i]->mem == nullptr) {
			*why = "cpu fallback: a binding is not in a CPU buffer";
			return false;
		}
		b[i].mem = d.bind[i]->mem;
		b[i].bytes = d.bind[i]->size;
	}
	if (!run_kernel(d.kernel, d.w, b, d.groups)) {
		*why = std::string("cpu fallback: kernel ") + kernel_desc(d.kernel).name + " has no cpp emit";
		return false;
	}
	return true;
}

} // namespace ggml_rd
