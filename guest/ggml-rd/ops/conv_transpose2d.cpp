// ggml-rd op: CONV_TRANSPOSE_2D (ggml_conv_transpose_2d_p0), f16 or f32
// kernel, f32 input, f32 output; stride in op_params[0].
//
// The kernels are lean/Ggml/SlangCodegen/ConvTranspose2d.lean
// (conv_transpose_2d_f32, conv_transpose_2d_f16): one thread per dst
// element (ox, oy, oc), gathering every input pixel and kernel tap that
// ggml-cpu's col2im scatters into it, in ggml-cpu's order of accumulation
// (input row, then column; a channel dot each); the f16 kernel rounds the
// input to f16 first, as ggml-cpu does. Every operand through its strides.
//
// supports(): dst f32 [(SW-1)s + KW, (SH-1)s + KH, OC, 1]; src0 (kernel
// [KW, KH, OC, IC]) f16 or f32 with one-element nb0; src1 (input
// [SW, SH, IC, 1]) f32 with nb0 = 4: ggml-cpu's own assertions, and its
// loops ignore an input batch beyond the first, so ne13 = 1; stride >= 1.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_conv_transpose2d(const ggml_tensor *op) {
	const ggml_tensor *k = op->src[0];
	const ggml_tensor *in = op->src[1];
	if (k == nullptr || in == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32 ||
			(k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16)) {
		return false;
	}
	if (k->nb[0] != ggml_type_size(k->type) || in->nb[0] != sizeof(float)) {
		return false;
	}
	const int64_t s = op->op_params[0];
	if (s < 1 || k->ne[3] != in->ne[2] || in->ne[3] != 1 || op->ne[3] != 1 || op->ne[2] != k->ne[2]) {
		return false;
	}
	if (op->ne[0] != (in->ne[0] - 1) * s + k->ne[0] || op->ne[1] != (in->ne[1] - 1) * s + k->ne[1]) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(k) && tensor_fits(in);
}

bool pack_conv_transpose2d(Pack &p) {
	p.kernel = kernel_index(p.node->src[0]->type == GGML_TYPE_F16 ? "conv_transpose_2d_f16" : "conv_transpose_2d_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(conv_transpose2d, GGML_OP_CONV_TRANSPOSE_2D, -1, supports_conv_transpose2d, pack_conv_transpose2d);
