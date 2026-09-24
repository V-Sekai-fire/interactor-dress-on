// ggml-rd op: CONV_2D_DW (ggml_conv_2d_dw_direct, depthwise), f32.
//
// The kernel is lean/Ggml/SlangCodegen/ConvDw.lean (conv_2d_dw_f32): one
// thread per dst element (ox, oy, c, n), the sum over the taps of channel
// c's kernel in ggml-cpu's order, every operand read through its strides:
// the WHCN and the channel-contiguous CWHN layouts (ggml gives dst the
// input's) are the one kernel. op_params words 37-42: s0 s1 p0 p1 d0 d1,
// read by the kernel; no derived words beyond the 1-D grid.
//
// supports(): src0 (the kernel) f32 [KW, KH, 1, C], src1 (the input) f32
// [W, H, C, N], dst f32 [OW, OH, C, N]; positive stride and dilation,
// non-negative padding (the kernel's padding test is unsigned); every
// extent and stride in the words and the tap coordinates in a word. The
// f16 kernel ggml-cpu also takes is not supported.
#include "../rd_pack.h"

namespace {

using namespace ggml_rd;

bool supports_conv_2d_dw_f32(const ggml_tensor *op) {
	const ggml_tensor *k = op->src[0];
	const ggml_tensor *img = op->src[1];
	if (k == nullptr || img == nullptr || op->src[2] != nullptr) {
		return false;
	}
	if (op->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || img->type != GGML_TYPE_F32) {
		return false;
	}
	if (k->ne[2] != 1 || k->ne[3] != img->ne[2] || op->ne[2] != img->ne[2] || op->ne[3] != img->ne[3]) {
		return false;
	}
	const int32_t *pp = reinterpret_cast<const int32_t *>(op->op_params);
	const int32_t s0 = pp[0], s1 = pp[1], p0 = pp[2], p1 = pp[3], d0 = pp[4], d1 = pp[5];
	if (s0 < 1 || s1 < 1 || p0 < 0 || p1 < 0 || d0 < 1 || d1 < 1) {
		return false;
	}
	if (uint64_t(op->ne[0] - 1) * uint64_t(s0) + uint64_t(k->ne[0] - 1) * uint64_t(d0) > UINT32_MAX ||
			uint64_t(op->ne[1] - 1) * uint64_t(s1) + uint64_t(k->ne[1] - 1) * uint64_t(d1) > UINT32_MAX) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(k) && tensor_fits(img);
}

bool pack_conv_2d_dw_f32(Pack &p) {
	p.kernel = kernel_index("conv_2d_dw_f32");
	return p.kernel >= 0 && grid_1d(p, uint64_t(ggml_nelements(p.node)));
}

} // namespace

GGML_RD_OP(conv_2d_dw_f32, GGML_OP_CONV_2D_DW, -1, supports_conv_2d_dw_f32, pack_conv_2d_dw_f32);
