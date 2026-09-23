// ggml-rd packing: from a ggml node to its params slot and its grid. Pure
// C++ over ggml tensors and the params words; no RenderingDevice, so the
// same code runs in the guest backend (rd_graph.cpp) and in the host test
// harness (tests/ggml_rd_kernels), which checks every packer against
// ggml-cpu through the kernels' slangc cpp emits.
//
// An op family adds ops/<op>.cpp with a supports() and a pack() and
// registers them with GGML_RD_OP; nothing else in guest/ggml-rd changes.
// Before pack() runs, fill_standard() has written every standard word: the
// dst block and one block per source (ne, nb and offset, in elements; in
// bytes for a block-quantized type) and the raw op_params. pack() picks the
// kernel (kernel_index("name")), fills the derived words 53..63 and the grid
// (grid_1d() for one thread per element).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml_rd_params.h"

#include "GgmlKernelTable.inc"

namespace ggml_rd {

struct Pack {
	const ggml_tensor *node = nullptr;
	uint32_t *w = nullptr; // this dispatch's 64 words
	int kernel = -1; // set by pack()
	uint32_t groups[3] = { 0, 0, 0 };
};

using SupportsFn = bool (*)(const ggml_tensor *op);
using PackFn = bool (*)(Pack &p);

struct OpEntry {
	const char *name;
	ggml_op op;
	int sub; // the ggml_unary_op of a UNARY, the ggml_glu_op of a GLU, else -1
	SupportsFn supports;
	PackFn pack;
};

void register_op(const OpEntry &e);
// The first registered entry for op's (op, sub) whose supports() accepts it.
const OpEntry *find_op(const ggml_tensor *op);
std::vector<const OpEntry *> ops();

struct OpRegistrar {
	explicit OpRegistrar(const OpEntry &e) { register_op(e); }
};

// Ops that only change a tensor's view of memory: never dispatched.
bool is_layout_only(ggml_op op);

int kernel_index(const char *name); // -1 if no such kernel
const KernelDesc &kernel_desc(int k);

// A 1-D launch of `threads` threads of the kernel's thread-group size,
// split as (gx, gy, 1) with gx, gy <= 65535; fills W_THREADS and W_GROUPS_X.
bool grid_1d(Pack &p, uint64_t threads);

// t's shape and element strides fit the uint32 params words, and its type
// has one element per block (f32, f16, bf16, i32...). No offset check: an
// unallocated tensor has none yet (fill_standard checks it).
bool tensor_fits(const ggml_tensor *t);

// Where a tensor's data starts in the buffer that is bound for it, in bytes.
using OffsetFn = bool (*)(const ggml_tensor *t, void *user, uint64_t *byte_offset);

// Zero the slot and write the standard words of `node`: word 0 is left for
// the kernel id. False, with a reason, if a tensor does not fit the words.
bool fill_standard(uint32_t *w, const ggml_tensor *node, OffsetFn offset, void *user, std::string *why);

} // namespace ggml_rd

// Register an op packer from ops/<op>.cpp (the guest links ggml_rd as an
// OBJECT library, so every registrar runs).
#define GGML_RD_OP(ident, op, sub, supports, pack) \
	static const ::ggml_rd::OpRegistrar ggml_rd_op_##ident({ #ident, (op), (sub), (supports), (pack) })
