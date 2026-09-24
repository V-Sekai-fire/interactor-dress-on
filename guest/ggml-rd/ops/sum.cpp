// ggml-rd ops: SUM and SUM_ROWS, f32.
//
// The kernels are lean/Ggml/SlangCodegen/Reduce.lean (sum_f32, sum_rows_f32
// and their _serial siblings for the host tests and the CPU fallback), f32
// sums tree-reduced in group-shared memory (ggml-cpu sums in double, so
// they match within test-backend-ops' NMSE, as MEAN does):
//
//   SUM_ROWS  a row kernel (ops/rows.h): one 256-thread group per row of
//             src0, dst f32 [1, ne1, ne2, ne3]; word 55 = rows.
//   SUM       one 256-thread group over every element of src0 (unravelled
//             over its ne, read through its strides, so a permuted source
//             is fine), dst one f32; word 53 = the kernel's thread count,
//             so the grid is that one group (grid_1d over tg threads).
//
// supports(): src0 and dst f32, one source, every stride in the words; for
// SUM the element count must fit a word (the kernel's n).
#include "rows.h"

namespace {

using namespace ggml_rd;

bool supports_sum_rows(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	return rows_f32_unary(op) && op->ne[0] == 1 && op->ne[1] == a->ne[1] && op->ne[2] == a->ne[2] &&
			op->ne[3] == a->ne[3];
}

bool supports_sum(const ggml_tensor *op) {
	return rows_f32_unary(op) && ggml_nelements(op) == 1 && ggml_nelements(op->src[0]) <= 0xFFFFFFFFll;
}

bool pack_sum_rows(Pack &p) {
	return pack_rows(p, "sum_rows_f32");
}

bool pack_sum(Pack &p) {
	p.kernel = kernel_index("sum_f32");
	if (p.kernel < 0) {
		return false;
	}
	return grid_1d(p, kernel_desc(p.kernel).threadgroup[0]);
}

} // namespace

GGML_RD_OP(sum_rows_f32, GGML_OP_SUM_ROWS, -1, supports_sum_rows, pack_sum_rows);
GGML_RD_OP(sum_f32, GGML_OP_SUM, -1, supports_sum, pack_sum);
