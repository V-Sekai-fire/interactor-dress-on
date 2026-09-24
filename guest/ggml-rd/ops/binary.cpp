// ggml-rd ops: ADD, SUB and MUL, f32, with ggml's broadcast. The reference packer
// (family K2); every other op file follows its shape:
//
//   supports(op)  -- exactly the cases the kernel computes: here all three
//                    tensors f32, dst the shape of src0, src1 repeatable into
//                    src0 (ggml_can_repeat), and every stride and extent
//                    expressible in the uint32 params words;
//   pack(p)       -- choose the kernel by its kernels.txt name, fill the
//                    derived words and the grid (the core has already written
//                    the dst/src blocks and op_params);
//   GGML_RD_OP    -- register both, once per (op, sub-op) this file serves.
//
// The kernels are lean/Ggml/SlangCodegen/Binary.lean (add_f32, sub_f32,
// mul_f32): one thread per dst element, dst[i] = s0[i] OP s1[i mod ne1],
// every operand strided. No derived words beyond the 1-D grid. SUB has
// ADD's broadcast rule (ggml_compute_forward_sub_f32 asserts
// ggml_can_repeat(src1, src0)), so the one supports() serves all three.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_binary_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || b == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(a, op) || !ggml_can_repeat(b, a)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(b);
}

const char *kernel_for(ggml_op op) {
	switch (op) {
		case GGML_OP_ADD:
			return "add_f32";
		case GGML_OP_SUB:
			return "sub_f32";
		default:
			return "mul_f32";
	}
}

bool pack_binary_f32(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node->op));
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(add_f32, GGML_OP_ADD, -1, supports_binary_f32, pack_binary_f32);
GGML_RD_OP(sub_f32, GGML_OP_SUB, -1, supports_binary_f32, pack_binary_f32);
GGML_RD_OP(mul_f32, GGML_OP_MUL, -1, supports_binary_f32, pack_binary_f32);
