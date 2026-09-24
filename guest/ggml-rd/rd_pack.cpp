#include "rd_pack.h"

#include <cstdlib>
#include <cstring>

namespace ggml_rd {

namespace {
std::vector<OpEntry> &op_table() {
	static std::vector<OpEntry> t;
	return t;
}
} // namespace

void register_op(const OpEntry &e) {
	op_table().push_back(e);
}

std::vector<const OpEntry *> ops() {
	std::vector<const OpEntry *> r;
	for (const OpEntry &e : op_table()) {
		r.push_back(&e);
	}
	return r;
}

const OpEntry *find_op(const ggml_tensor *op) {
	int sub = -1;
	if (op->op == GGML_OP_UNARY) {
		sub = int(ggml_get_unary_op(op));
	} else if (op->op == GGML_OP_GLU) {
		sub = int(ggml_get_glu_op(op));
	}
	for (const OpEntry &e : op_table()) {
		if (e.op == op->op && (e.sub == -1 || e.sub == sub) && e.supports(op)) {
			return &e;
		}
	}
	return nullptr;
}

bool is_layout_only(ggml_op op) {
	return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
			op == GGML_OP_TRANSPOSE;
}

static bool g_serial = false;

void set_serial_kernels(bool on) {
	g_serial = on;
}

bool serial_kernels() {
	if (g_serial) {
		return true;
	}
	const char *e = std::getenv("GGML_RD_SERIAL");
	return e != nullptr && std::atoi(e) != 0;
}

int kernel_index(const char *name) {
	for (uint32_t i = 0; i < kKernelCount; ++i) {
		if (std::strcmp(kKernels[i].name, name) == 0) {
			return int(i);
		}
	}
	return -1;
}

const KernelDesc &kernel_desc(int k) {
	return kKernels[k];
}

bool grid_1d(Pack &p, uint64_t threads) {
	if (p.kernel < 0 || uint32_t(p.kernel) >= kKernelCount || threads == 0 || threads > 0xFFFFFFFFull) {
		return false;
	}
	const uint64_t tg = kKernels[p.kernel].threadgroup[0];
	const uint64_t groups = (threads + tg - 1) / tg;
	const uint64_t gx = groups < kMaxGroupsPerDim ? groups : kMaxGroupsPerDim;
	const uint64_t gy = (groups + gx - 1) / gx;
	if (gy > kMaxGroupsPerDim) {
		return false;
	}
	p.w[W_THREADS] = uint32_t(threads);
	p.w[W_GROUPS_X] = uint32_t(gx);
	p.groups[0] = uint32_t(gx);
	p.groups[1] = uint32_t(gy);
	p.groups[2] = 1;
	return true;
}

bool tensor_fits(const ggml_tensor *t) {
	if (t == nullptr) {
		return true;
	}
	if (ggml_blck_size(t->type) != 1) {
		return false;
	}
	const size_t ts = ggml_type_size(t->type);
	uint64_t last = 0; // the largest element offset the tensor reaches
	for (int k = 0; k < 4; ++k) {
		if (t->ne[k] < 1 || uint64_t(t->ne[k]) > 0xFFFFFFFFull || t->nb[k] % ts != 0) {
			return false;
		}
		const uint64_t nb = t->nb[k] / ts;
		if (nb > 0xFFFFFFFFull) {
			return false;
		}
		last += uint64_t(t->ne[k] - 1) * nb;
	}
	return last <= 0xFFFFFFFFull && uint64_t(ggml_nelements(t)) <= 0xFFFFFFFFull;
}

namespace {

// One tensor block: ne, nb and offset, in elements (in bytes for a
// block-quantized type).
bool fill_block(uint32_t *w, uint32_t at, const ggml_tensor *t, uint64_t off) {
	const size_t ts = ggml_type_size(t->type);
	const bool elems = ggml_blck_size(t->type) == 1;
	for (int k = 0; k < 4; ++k) {
		if (t->ne[k] < 0 || uint64_t(t->ne[k]) > 0xFFFFFFFFull) {
			return false;
		}
		w[at + T_NE + k] = uint32_t(t->ne[k]);
		uint64_t nb = uint64_t(t->nb[k]);
		if (elems) {
			if (nb % ts != 0) {
				return false;
			}
			nb /= ts;
		}
		if (nb > 0xFFFFFFFFull) {
			return false;
		}
		w[at + T_NB + k] = uint32_t(nb);
	}
	if (elems) {
		if (off % ts != 0) {
			return false;
		}
		off /= ts;
	}
	if (off > 0xFFFFFFFFull) {
		return false;
	}
	w[at + T_OFF] = uint32_t(off);
	return true;
}

} // namespace

bool fill_standard(uint32_t *w, const ggml_tensor *node, OffsetFn offset, void *user, std::string *why) {
	std::memset(w, 0, kSlotBytes);
	uint64_t off = 0;
	if (!offset(node, user, &off) || !fill_block(w, W_DST, node, off)) {
		if (why) {
			*why = "the destination is not in a bound buffer, or does not fit the params words";
		}
		return false;
	}
	static const uint32_t kSrcWord[3] = { W_SRC0, W_SRC1, W_SRC2 };
	for (int s = 0; s < 3; ++s) {
		const ggml_tensor *src = node->src[s];
		if (src == nullptr) {
			continue;
		}
		if (!offset(src, user, &off) || !fill_block(w, kSrcWord[s], src, off)) {
			if (why) {
				*why = "source " + std::to_string(s) + " is not in a bound buffer, or does not fit the params words";
			}
			return false;
		}
	}
	static_assert(sizeof(node->op_params) == N_OP_PARAMS * 4, "op_params is 16 words");
	std::memcpy(w + W_OP_PARAMS, node->op_params, sizeof(node->op_params));
	return true;
}

} // namespace ggml_rd
