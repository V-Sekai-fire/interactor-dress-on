// ggml-rd op: PAD, f32 (ggml_pad, ggml_pad_ext, ggml_pad_ext_circular).
//
// The kernel is lean/Ggml/SlangCodegen/Pad.lean (pad_f32): one thread per
// dst element; zero fill outside [lp, lp + src0.ne) in each dimension, or,
// with op_params[8] set, the source wrapped around (ggml-cpu's
// ggml_compute_forward_pad_f32<false> and <true>). Both operands strided.
// op_params words 0..7 are lp0 rp0 lp1 rp1 lp2 rp2 lp3 rp3, read raw.
//
// supports(): src0 and dst f32, one source, dst.ne = src0.ne + lp + rp per
// dimension with every pad >= 0 (the kernel's tests are unsigned), and for
// the circular form lp <= src0.ne (where ggml-cpu's own (coord + size) %
// size stays non-negative).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_pad_f32(const ggml_tensor *op) {
	const ggml_tensor *a = op->src[0];
	if (a == nullptr || op->src[1] != nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
		return false;
	}
	const int32_t *pp = op->op_params;
	const bool circular = pp[8] != 0;
	for (int k = 0; k < 4; ++k) {
		const int64_t lp = pp[2 * k], rp = pp[2 * k + 1];
		if (lp < 0 || rp < 0 || op->ne[k] != a->ne[k] + lp + rp) {
			return false;
		}
		if (circular && lp > a->ne[k]) {
			return false;
		}
	}
	return tensor_fits(op) && tensor_fits(a);
}

bool pack_pad_f32(Pack &p) {
	p.kernel = kernel_index("pad_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(pad_f32, GGML_OP_PAD, -1, supports_pad_f32, pack_pad_f32);
