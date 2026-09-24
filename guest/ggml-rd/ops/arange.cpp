// ggml-rd op: ARANGE, f32 (ggml_arange: start, stop, step as floats in
// op_params words 0..2).
//
// The kernel is lean/Ggml/SlangCodegen/Arange.lean (arange_f32): one thread
// per element, dst[i] = start + step * float(i), written through dst's
// offset and stride. No source.
//
// supports(): dst f32 and 1-D (ggml_arange makes it so), no source, step
// non-zero, and ceilf((stop - start) / step) elements, which ggml asserts
// in the op and would abort on.
#include <cmath>
#include <cstring>

#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_arange_f32(const ggml_tensor *op) {
	if (op->src[0] != nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
		return false;
	}
	float start, stop, step; // op_params hold them bit for bit (ggml_set_op_params_f32)
	std::memcpy(&start, op->op_params + 0, 4);
	std::memcpy(&stop, op->op_params + 1, 4);
	std::memcpy(&step, op->op_params + 2, 4);
	if (!(step != 0.0f) || !std::isfinite(start) || !std::isfinite(stop) || !std::isfinite(step)) {
		return false;
	}
	const double steps = std::ceil(double((stop - start) / step));
	if (!(steps >= 1.0) || steps != double(ggml_nelements(op))) {
		return false;
	}
	return tensor_fits(op);
}

bool pack_arange_f32(Pack &p) {
	p.kernel = kernel_index("arange_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(arange_f32, GGML_OP_ARANGE, -1, supports_arange_f32, pack_arange_f32);
