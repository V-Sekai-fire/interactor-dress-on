// ggml-rd ops: IM2COL (family K7), 2-D and 1-D, f32 image -> f32 or f16
// columns. The kernels are lean/Ggml/SlangCodegen/Conv.lean:
//
//   im2col_f32  one thread per dst element, dst through its strides;
//   im2col_f16  one thread per 32-bit word of a contiguous dst: the two
//               halves in it, rounded to nearest even as ggml-cpu rounds
//               (a half outside dst keeps the word's old bits).
//
// supports() is exactly what ggml-cpu's im2col computes the same way: src1
// (the image) f32 with nb10 = 4, rows contiguous in 2-D (ggml-cpu indexes a
// channel as src[iih*IW + iiw]); src0 (the kernel) is read for its shape
// only, any one-element-per-block type; positive stride and dilation,
// non-negative padding (the kernel's padding test is unsigned). No
// derived words beyond the 1-D grid: the kernels read the shapes, strides
// and op_params from the standard words.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_im2col(const ggml_tensor *op) {
	const ggml_tensor *k = op->src[0];
	const ggml_tensor *img = op->src[1];
	if (k == nullptr || img == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (img->type != GGML_TYPE_F32 || (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16)) {
		return false;
	}
	const int32_t *pp = reinterpret_cast<const int32_t *>(op->op_params);
	const int32_t s0 = pp[0], s1 = pp[1], p0 = pp[2], p1 = pp[3], d0 = pp[4], d1 = pp[5], is2d = pp[6];
	if (is2d != 0 && is2d != 1) {
		return false;
	}
	if (s0 < 1 || d0 < 1 || p0 < 0 || (is2d && (s1 < 1 || d1 < 1 || p1 < 0))) {
		return false;
	}
	if (img->nb[0] != sizeof(float)) {
		return false;
	}
	if (is2d) {
		if (img->nb[1] != img->ne[0] * int64_t(sizeof(float)) || k->ne[2] != img->ne[2]) {
			return false;
		}
	} else if (k->ne[1] != img->ne[1] || img->ne[3] != 1) {
		return false;
	}
	// The f16 kernel owns whole 32-bit words of a contiguous dst.
	if (op->type == GGML_TYPE_F16 && !ggml_is_contiguous(op)) {
		return false;
	}
	if (ggml_nelements(op) >= (int64_t(1) << 31)) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(k) && tensor_fits(img);
}

bool pack_im2col(Pack &p) {
	if (p.node->type == GGML_TYPE_F32) {
		p.kernel = kernel_index("im2col_f32");
		return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
	}
	// f16: one thread per word from the one holding dst's first half to the
	// one holding its last.
	p.kernel = kernel_index("im2col_f16");
	const uint64_t off = p.w[W_DST + T_OFF];
	const uint64_t n = uint64_t(ggml_nelements(p.node));
	const uint64_t words = ((off + n - 1) >> 1) - (off >> 1) + 1;
	return p.kernel >= 0 && grid_1d(p, words);
}

} // namespace

GGML_RD_OP(im2col, GGML_OP_IM2COL, -1, supports_im2col, pack_im2col);
