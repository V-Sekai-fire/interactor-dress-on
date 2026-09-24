// ggml-rd op: GROUP_NORM, f32 (ggml_group_norm: n_groups in op_params[0],
// eps in op_params[1]).
//
// The kernel is lean/Ggml/SlangCodegen/GroupNorm.lean (group_norm_f32, and
// group_norm_f32_serial for the host tests, which gen_host_kernels.py runs
// in its place as it does the NORM siblings): one 256-thread work group per
// (group, batch) of src0's dim-2 channels cut into n_groups groups of
// ceil(ne2 / n_groups) (the last shorter, or empty), the group's ne0 * ne1
// * step elements summed grid-strided and tree-reduced, dst = (x - mean) *
// 1 / sqrt(variance + eps). Every operand strided (ggml-cpu asserts nb0 = 4
// only). Derived word 55 = rows = n_groups * ne3, the work groups.
//
// supports(): src0 and dst f32 of one shape, one source, n_groups >= 1, a
// group's element count and the row count in the uint32 words.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

constexpr uint32_t W_ROWS = W_DERIVED + 2;
static_assert(W_ROWS == 55, "Rows.lean: word 55 is the row count");

bool supports_group_norm_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, op)) {
		return false;
	}
	const int64_t ng = op->op_params[0];
	if (ng < 1) {
		return false;
	}
	const uint64_t cpg = (uint64_t(a->ne[2]) + uint64_t(ng) - 1) / uint64_t(ng);
	if (uint64_t(a->ne[0]) * uint64_t(a->ne[1]) * cpg > 0xFFFFFFFFull) {
		return false;
	}
	if (uint64_t(ng) * uint64_t(a->ne[3]) > 0xFFFFFFFFull / 256) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_group_norm_f32(Pack &p) {
	p.kernel = kernel_index("group_norm_f32");
	if (p.kernel < 0) {
		return false;
	}
	const uint64_t rows = uint64_t(p.node->op_params[0]) * uint64_t(p.node->src[0]->ne[3]);
	const uint64_t tg = kernel_desc(p.kernel).threadgroup[0];
	p.w[W_ROWS] = uint32_t(rows);
	return grid_1d(p, rows * tg);
}

} // namespace

GGML_RD_OP(group_norm_f32, GGML_OP_GROUP_NORM, -1, supports_group_norm_f32, pack_group_norm_f32);
