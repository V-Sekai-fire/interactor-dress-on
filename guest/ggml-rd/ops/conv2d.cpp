// ggml-rd op: CONV_2D (ggml_conv_2d_direct, family K7), f32 or f16 kernel,
// f32 input, f32 output. The kernels are lean/Ggml/SlangCodegen/Conv2d.lean
// (conv2d_f32, conv2d_f16): one thread per dst element, the IC x KH x KW
// window summed in f32 in ggml-cpu's im2col order (ic, ky, kx), every
// operand through its strides; conv2d_f16 rounds the input to f16 first,
// as ggml-cpu does (its patch is in the kernel's type). ops/conv3d.cpp
// without the depth.
//
// supports(): dst f32 [OW, OH, OC, N], src0 (kernel [KW, KH, IC, OC]) f32 or
// f16 and contiguous (ggml-cpu asserts it), src1 (input [W, H, IC, N]) f32,
// IC and OC and N consistent between the three, positive stride and
// dilation, non-negative padding (the kernel's bounds test is unsigned);
// op_params s0 s1 p0 p1 d0 d1 read raw by the kernel.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_conv2d(const ggml_tensor *op) {
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
	const int32_t *pp = op->op_params;
	for (int i = 0; i < 2; ++i) {
		if (pp[i] < 1 || pp[2 + i] < 0 || pp[4 + i] < 1) {
			return false;
		}
	}
	if (k->ne[2] != in->ne[2] || op->ne[2] != k->ne[3] || op->ne[3] != in->ne[3]) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(k) && tensor_fits(in);
}

bool pack_conv2d(Pack &p) {
	p.kernel = kernel_index(p.node->src[0]->type == GGML_TYPE_F16 ? "conv2d_f16" : "conv2d_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(conv2d, GGML_OP_CONV_2D, -1, supports_conv2d, pack_conv2d);
