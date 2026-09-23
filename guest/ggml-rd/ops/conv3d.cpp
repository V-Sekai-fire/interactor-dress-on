// ggml-rd ops: CONV_3D (ggml_conv_3d_direct, family K7), f32 or f16 kernel,
// f32 input, f32 output. The kernels are lean/Ggml/SlangCodegen/Conv.lean
// (conv3d_f32, conv3d_f16): one thread per dst element, the whole
// IC x KD x KH x KW window summed in f32 in ggml-cpu's order, every operand
// through its strides; conv3d_f16 rounds the input to f16 first, as ggml-cpu
// does (its im2col is in the kernel's type).
//
// supports(): dst f32, src0 (kernel [KW,KH,KD,IC*OC]) f32 or f16 and
// contiguous (ggml-cpu asserts it), src1 (input [W,H,D,IC*N]) f32, the
// op_params' c, n and oc consistent with the shapes, positive stride and
// dilation, non-negative padding (the kernel's bounds test is unsigned).
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_conv3d(const ggml_tensor *op) {
	const ggml_tensor *k = op->src[0];
	const ggml_tensor *in = op->src[1];
	if (k == nullptr || in == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32 ||
			(k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16)) {
		return false;
	}
	if (!ggml_is_contiguous(k)) {
		return false;
	}
	const int32_t *pp = reinterpret_cast<const int32_t *>(op->op_params);
	for (int i = 0; i < 3; ++i) {
		if (pp[i] < 1 || pp[3 + i] < 0 || pp[6 + i] < 1) {
			return false;
		}
	}
	const int64_t c = pp[9], n = pp[10], oc = pp[11];
	if (c < 1 || n < 1 || oc < 1 || k->ne[3] != c * oc || in->ne[3] != c * n || op->ne[3] != oc * n) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(k) && tensor_fits(in);
}

bool pack_conv3d(Pack &p) {
	p.kernel = kernel_index(p.node->src[0]->type == GGML_TYPE_F16 ? "conv3d_f16" : "conv3d_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(conv3d, GGML_OP_CONV_3D, -1, supports_conv3d, pack_conv3d);
