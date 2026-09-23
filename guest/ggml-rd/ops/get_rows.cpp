// ggml-rd op: GET_ROWS, {f32, i32, f16, bf16} rows gathered by i32 indices.
//
// The kernels are lean/Ggml/SlangCodegen/GetRows.lean:
// dst[i0, i1, i2, i3] = src0[i0, rows[i1, i2, i3], i2, i3], one thread per
// dst element in dst's own order, every operand strided. dst is f32 (i32 for
// an i32 src0), as ggml_get_rows makes it.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

const char *kernel_for(ggml_type s, ggml_type d) {
	if ((s == GGML_TYPE_F32 && d == GGML_TYPE_F32) || (s == GGML_TYPE_I32 && d == GGML_TYPE_I32)) return "get_rows_b32";
	if (s == GGML_TYPE_F16 && d == GGML_TYPE_F32) return "get_rows_f16";
	if (s == GGML_TYPE_BF16 && d == GGML_TYPE_F32) return "get_rows_bf16";
	return nullptr;
}

bool supports_get_rows(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *r = op->src[1];
	if (a == nullptr || r == nullptr || op->src[2] != nullptr || r->type != GGML_TYPE_I32) {
		return false;
	}
	if (kernel_for(a->type, op->type) == nullptr) {
		return false;
	}
	// ggml_get_rows' shapes: dst [a.ne0, r.ne0, r.ne1, r.ne2], a.ne2 = r.ne1, a.ne3 = r.ne2.
	if (op->ne[0] != a->ne[0] || op->ne[1] != r->ne[0] || op->ne[2] != r->ne[1] || op->ne[3] != r->ne[2] ||
			a->ne[2] != r->ne[1] || a->ne[3] != r->ne[2] || r->ne[3] != 1) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(a) && tensor_fits(r);
}

bool pack_get_rows(Pack &p) {
	p.kernel = kernel_index(kernel_for(p.node->src[0]->type, p.node->type));
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(get_rows, GGML_OP_GET_ROWS, -1, supports_get_rows, pack_get_rows);
