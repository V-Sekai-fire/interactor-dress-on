// ggml-rd op: SET, f32 (ggml_set, ggml_set_1d, ggml_set_2d, in place or
// not): dst is src0 with src1 written into the window at byte `offset`
// with row strides nb1, nb2, nb3 (op_params words 37-40; word 41 is the
// inplace flag, which changes nothing here: every dst element is written).
//
// The kernel is lean/Ggml/SlangCodegen/Set.lean (set_f32): one thread per
// dst element selects src1's element when its linear index falls in the
// window, else src0's. It recovers the window index from the linear index
// by dividing by the strides, which is exact only for a nested window:
// nb1 >= ne10, nb2 >= ne11 * nb1, nb3 >= ne12 * nb2 (in elements). Derived
// words 55-58 are nb1, nb2, nb3 and offset in elements.
//
// supports(): src0 and dst f32 of one shape, both contiguous (ggml-cpu
// asserts it; the linear index is the element offset); src1 f32 with any
// strides; the strides and offset multiples of 4 bytes, positive, nested,
// and the window inside dst (ggml asserts it); everything in the words.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

constexpr uint32_t W_NB1 = W_DERIVED + 2; // Set.wNb1
constexpr uint32_t W_OFFSET = W_DERIVED + 5; // Set.wOffset
static_assert(W_NB1 == 55 && W_OFFSET == 58, "Set.lean: words 55-58");

struct Window {
	uint64_t nb[3], off; // in elements
};

bool window_of(const ggml_tensor *op, Window &w) {
	const ggml_tensor *b = op->src[1];
	const int32_t *pp = reinterpret_cast<const int32_t *>(op->op_params);
	for (int k = 0; k < 4; ++k) {
		if (pp[k] < 0 || pp[k] % 4 != 0) {
			return false;
		}
	}
	for (int k = 0; k < 3; ++k) {
		w.nb[k] = uint64_t(pp[k]) / 4;
	}
	w.off = uint64_t(pp[3]) / 4;
	if (w.nb[0] < uint64_t(b->ne[0]) || w.nb[1] < uint64_t(b->ne[1]) * w.nb[0] ||
			w.nb[2] < uint64_t(b->ne[2]) * w.nb[1]) {
		return false;
	}
	const uint64_t last = w.off + uint64_t(b->ne[0] - 1) + uint64_t(b->ne[1] - 1) * w.nb[0] +
			uint64_t(b->ne[2] - 1) * w.nb[1] + uint64_t(b->ne[3] - 1) * w.nb[2];
	return last < uint64_t(ggml_nelements(op)) && w.nb[2] <= UINT32_MAX && w.off <= UINT32_MAX;
}

bool supports_set_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	const ggml_tensor *b = op->src[1];
	if (a == nullptr || b == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
		return false;
	}
	if (!ggml_are_same_shape(a, op) || !ggml_is_contiguous(op) || !ggml_is_contiguous(a)) {
		return false;
	}
	Window w;
	return window_of(op, w) && tensor_fits(op) && tensor_fits(a) && tensor_fits(b);
}

bool pack_set_f32(Pack &p) {
	p.kernel = kernel_index("set_f32");
	Window w;
	if (p.kernel < 0 || !window_of(p.node, w)) {
		return false;
	}
	for (int k = 0; k < 3; ++k) {
		p.w[W_NB1 + k] = uint32_t(w.nb[k]);
	}
	p.w[W_OFFSET] = uint32_t(w.off);
	return grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(set_f32, GGML_OP_SET, -1, supports_set_f32, pack_set_f32);
