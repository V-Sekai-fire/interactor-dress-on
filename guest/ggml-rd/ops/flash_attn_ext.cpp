// ggml-rd op: FLASH_ATTN_EXT without a mask (family K8).
//
//   supports(op)  -- Q f32; K and V both f32 or both f16; head size
//                    D = 64 or 128 for Q, K, V and dst; no mask (src[3]) and
//                    no sinks (src[4]); K and V the same length; grouped-query
//                    broadcast (Hq a multiple of Hk and Hv, B of Bk and Bv);
//                    rows contiguous (nb0 = the type size, as ggml-cpu
//                    asserts); dst f32 [D, Hq, N, B]; precision DEFAULT or
//                    F32 (both accumulate in f32); a softcap is computed, and
//                    max_bias is accepted because without a mask ALiBi has
//                    nothing to scale (ggml-cpu multiplies the mask by it).
//   pack(p)       -- the tiled kernel (16 queries of one head per group of
//                    128 threads, grid (ceil(N / 16), Hq, B)), or under
//                    GGML_RD_SERIAL=1 its _serial sibling (one thread per
//                    query row, a 1-D grid): the host L2 harness runs the
//                    sibling, which has a cpp emit, and on the GPU it is the
//                    A/B for the tiled kernel. Derived words: 55-58 the
//                    broadcast ratios Hq/Hk, B/Bk, Hq/Hv, B/Bv; 59 the scale
//                    (divided by the softcap when there is one, exactly as
//                    ggml-cpu does it, in f32); 60 the softcap.
//
// The kernels are lean/Ggml/SlangCodegen/FlashAttn.lean.
#include "../rd_pack.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using namespace ggml_rd;

constexpr uint32_t kRows = 16; // queries per work group of the tiled kernel
constexpr uint32_t W_RK2 = 55, W_RK3 = 56, W_RV2 = 57, W_RV3 = 58, W_SCALE = 59, W_SOFTCAP = 60;

// op_params word i (ggml-impl.h's accessors are internal to ggml).
int32_t param_i32(const ggml_tensor *op, int i) {
	int32_t x;
	std::memcpy(&x, reinterpret_cast<const char *>(op->op_params) + 4 * i, 4);
	return x;
}

float param_f32(const ggml_tensor *op, int i) {
	float x;
	std::memcpy(&x, reinterpret_cast<const char *>(op->op_params) + 4 * i, 4);
	return x;
}

bool rows_contiguous(const ggml_tensor *t) {
	return t->nb[0] == ggml_type_size(t->type);
}

bool supports_fa(const ggml_tensor *op) {
	const ggml_tensor *q = op->src[0];
	const ggml_tensor *k = op->src[1];
	const ggml_tensor *v = op->src[2];
	if (q == nullptr || k == nullptr || v == nullptr || op->src[3] != nullptr || op->src[4] != nullptr) {
		return false; // a mask or sinks: not this kernel
	}
	if (q->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
		return false;
	}
	if (k->type != v->type || (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16)) {
		return false;
	}
	const int64_t d = q->ne[0];
	if ((d != 64 && d != 128) || k->ne[0] != d || v->ne[0] != d) {
		return false;
	}
	const int prec = param_i32(op, 3);
	if (prec != GGML_PREC_DEFAULT && prec != GGML_PREC_F32) {
		return false;
	}
	// ggml_flash_attn_ext's shapes: K and V [D, Lk, Hk, Bk], dst [D, Hq, N, B].
	if (k->ne[1] != v->ne[1] || k->ne[1] < 1) {
		return false;
	}
	if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || q->ne[3] % k->ne[3] != 0 ||
			q->ne[3] % v->ne[3] != 0) {
		return false;
	}
	if (op->ne[0] != d || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1] || op->ne[3] != q->ne[3]) {
		return false;
	}
	if (!rows_contiguous(q) || !rows_contiguous(k) || !rows_contiguous(v) || !rows_contiguous(op)) {
		return false;
	}
	// The tiled grid: query blocks, heads and batches each <= 65535 groups.
	const int64_t qblocks = (q->ne[1] + kRows - 1) / kRows;
	if (qblocks > kMaxGroupsPerDim || q->ne[2] > kMaxGroupsPerDim || q->ne[3] > kMaxGroupsPerDim) {
		return false;
	}
	return tensor_fits(op) && tensor_fits(q) && tensor_fits(k) && tensor_fits(v);
}

bool pack_fa(Pack &p) {
	const ggml_tensor *op = p.node;
	const ggml_tensor *q = op->src[0];
	const ggml_tensor *k = op->src[1];
	const ggml_tensor *v = op->src[2];
	const bool serial = serial_kernels();
	std::string name = std::string("flash_attn_ext_") + (k->type == GGML_TYPE_F16 ? "f16" : "f32") + "_d" +
			std::to_string(q->ne[0]) + (serial ? "_serial" : "");
	p.kernel = kernel_index(name.c_str());
	if (p.kernel < 0) {
		return false;
	}
	float scale = param_f32(op, 0);
	const float softcap = param_f32(op, 2);
	if (softcap != 0.0f) {
		scale /= softcap; // ggml-cpu: scale /= logit_softcap, then softcap * tanhf(s * scale)
	}
	p.w[W_RK2] = uint32_t(q->ne[2] / k->ne[2]);
	p.w[W_RK3] = uint32_t(q->ne[3] / k->ne[3]);
	p.w[W_RV2] = uint32_t(q->ne[2] / v->ne[2]);
	p.w[W_RV3] = uint32_t(q->ne[3] / v->ne[3]);
	std::memcpy(&p.w[W_SCALE], &scale, 4);
	std::memcpy(&p.w[W_SOFTCAP], &softcap, 4);
	if (serial) {
		return grid_1d(p, uint64_t(q->ne[1]) * uint64_t(q->ne[2]) * uint64_t(q->ne[3]));
	}
	p.groups[0] = uint32_t((q->ne[1] + kRows - 1) / kRows);
	p.groups[1] = uint32_t(q->ne[2]);
	p.groups[2] = uint32_t(q->ne[3]);
	return true;
}

} // namespace

GGML_RD_OP(flash_attn_ext, GGML_OP_FLASH_ATTN_EXT, -1, supports_fa, pack_fa);
