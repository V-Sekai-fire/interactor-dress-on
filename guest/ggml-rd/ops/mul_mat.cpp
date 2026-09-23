// ggml-rd ops: MUL_MAT (family K6). dst = src1 . src0^T per batch, f32 out:
//
//   src0 [K, M, ne02, ne03]  f32 | f16 | bf16      src1 [K, N, ne12, ne13]  f32 (or f16 with an f16 src0)
//   dst  [M, N, ne12, ne13]  f32, r2 = ne12 / ne02, r3 = ne13 / ne03 (ggml's batch broadcast)
//
// supports() accepts exactly the (src0, src1) pairs that have kernels:
// f32 x f32, f16 x f32, bf16 x f32 and f16 x f16 (the census's rows;
// ggml-cpu refuses f32 x f16 and bf16 x f16, so nothing could check them),
// any strides (every operand is read through its nb: permuted and
// non-contiguous src0/src1, strided dst), and a grid that fits 65535 groups
// per dimension. The sums are f32 whatever the inputs, so GGML_PREC_F32
// (op_params[0]) holds without being read; hints (op_params[1]) change
// nothing.
//
// pack() picks the kernel (lean/Ggml/SlangCodegen/MulMat*.lean):
//   N <= 4  mul_mat_vec_<a>_<b>    32 lanes x 8 rows per group; grid
//                                  (gx, ceil(ceil(M/8)/gx), ne12*ne13), gx in word 57
//   N > 4   mul_mat_tiled_<a>_<b>  16 x 16 threads, a 64 x 64 tile per group; grid
//                                  (ceil(M/64), ceil(N/64), ne12*ne13)
// and writes r2 (word 55) and r3 (word 56). The host harness runs each
// kernel's serial sibling (kernels/ggml/cpp_siblings.txt) over the same
// grid and words.
#include "../rd_pack.h"

#include <string>

namespace {

using namespace ggml_rd;

constexpr uint32_t W_R2 = 55;
constexpr uint32_t W_R3 = 56;
constexpr uint32_t W_ROW_GROUPS_X = 57;
constexpr uint64_t kVecMaxCols = 4;
constexpr uint64_t kVecRows = 8;
constexpr uint64_t kTile = 64;

const char *type_tag(ggml_type t) {
	switch (t) {
		case GGML_TYPE_F32:
			return "f32";
		case GGML_TYPE_F16:
			return "f16";
		case GGML_TYPE_BF16:
			return "bf16";
		default:
			return nullptr;
	}
}

bool has_kernel(ggml_type a, ggml_type b) {
	if (b == GGML_TYPE_F32) {
		return a == GGML_TYPE_F32 || a == GGML_TYPE_F16 || a == GGML_TYPE_BF16;
	}
	return b == GGML_TYPE_F16 && a == GGML_TYPE_F16;
}

uint64_t cdiv(uint64_t a, uint64_t b) {
	return (a + b - 1) / b;
}

// The grid for dst's (M, N, batches), or false if it does not fit.
bool grid(const ggml_tensor *op, uint32_t g[3], uint32_t *row_groups_x) {
	const uint64_t m = uint64_t(op->ne[0]);
	const uint64_t n = uint64_t(op->ne[1]);
	const uint64_t batches = uint64_t(op->ne[2]) * uint64_t(op->ne[3]);
	if (batches > kMaxGroupsPerDim) {
		return false;
	}
	if (n <= kVecMaxCols) {
		const uint64_t blocks = cdiv(m, kVecRows);
		const uint64_t gx = blocks < kMaxGroupsPerDim ? blocks : kMaxGroupsPerDim;
		const uint64_t gy = cdiv(blocks, gx);
		if (gy > kMaxGroupsPerDim) {
			return false;
		}
		g[0] = uint32_t(gx);
		g[1] = uint32_t(gy);
		*row_groups_x = uint32_t(gx);
	} else {
		const uint64_t gx = cdiv(m, kTile);
		const uint64_t gy = cdiv(n, kTile);
		if (gx > kMaxGroupsPerDim || gy > kMaxGroupsPerDim) {
			return false;
		}
		g[0] = uint32_t(gx);
		g[1] = uint32_t(gy);
		*row_groups_x = 0;
	}
	g[2] = uint32_t(batches);
	return true;
}

bool supports_mul_mat(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || b == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || !has_kernel(a->type, b->type)) {
		return false;
	}
	if (a->ne[0] != b->ne[0] || op->ne[0] != a->ne[1] || op->ne[1] != b->ne[1] || op->ne[2] != b->ne[2] ||
			op->ne[3] != b->ne[3]) {
		return false;
	}
	if (!tensor_fits(op) || !tensor_fits(a) || !tensor_fits(b)) {
		return false;
	}
	if (b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
		return false;
	}
	uint32_t g[3];
	uint32_t gx;
	return grid(op, g, &gx);
}

bool pack_mul_mat(Pack &p) {
	const ggml_tensor *a = p.node->src[0];
	const ggml_tensor *b = p.node->src[1];
	const bool vec = uint64_t(p.node->ne[1]) <= kVecMaxCols;
	const std::string name =
			std::string(vec ? "mul_mat_vec_" : "mul_mat_tiled_") + type_tag(a->type) + "_" + type_tag(b->type);
	p.kernel = kernel_index(name.c_str());
	if (p.kernel < 0) {
		return false;
	}
	uint32_t gx = 0;
	if (!grid(p.node, p.groups, &gx)) {
		return false;
	}
	p.w[W_R2] = uint32_t(b->ne[2] / a->ne[2]);
	p.w[W_R3] = uint32_t(b->ne[3] / a->ne[3]);
	p.w[W_ROW_GROUPS_X] = gx;
	return true;
}

} // namespace

GGML_RD_OP(mul_mat, GGML_OP_MUL_MAT, -1, supports_mul_mat, pack_mul_mat);
