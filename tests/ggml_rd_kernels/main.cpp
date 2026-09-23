// tests/ggml_rd_kernels: L2 of the ggml-rd kernel checks (l2.h says what).
//
//   ggml_rd_l2                    every case: the kernels vs ggml-cpu
//   ggml_rd_l2 --control=swap-nb  the same with src0's nb1 and nb2 swapped
//                                 after packing (src1's for a case with
//                                 control_src = 1): every case where the swap
//                                 changes an address must FAIL
//
// The last line is "RESULT: PASS" or "RESULT: FAIL".
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "l2.h"
#include "rd_pack.h"

bool run_kernel(int id, uint32_t *words, void *mem, size_t bytes, const uint32_t groups[3]);

static std::vector<L2Maker> &makers() {
	static std::vector<L2Maker> m;
	return m;
}

L2Registrar::L2Registrar(L2Maker m) {
	makers().push_back(m);
}

namespace {

using namespace ggml_rd;

struct Mem {
	uint8_t *base;
	size_t bytes;
};

bool host_offset(const ggml_tensor *t, void *user, uint64_t *off) {
	const Mem *m = static_cast<const Mem *>(user);
	const uint8_t *d = static_cast<const uint8_t *>(t->data);
	if (d < m->base || d >= m->base + m->bytes) {
		return false;
	}
	*off = uint64_t(d - m->base);
	return true;
}

float elem(const ggml_tensor *t, size_t off) {
	const uint8_t *p = static_cast<const uint8_t *>(t->data) + off;
	switch (t->type) {
		case GGML_TYPE_F32: {
			float f;
			std::memcpy(&f, p, 4);
			return f;
		}
		case GGML_TYPE_F16: {
			ggml_fp16_t h;
			std::memcpy(&h, p, 2);
			return ggml_fp16_to_fp32(h);
		}
		case GGML_TYPE_BF16: {
			ggml_bf16_t h;
			std::memcpy(&h, p, 2);
			return ggml_bf16_to_fp32(h);
		}
		case GGML_TYPE_I32: {
			int32_t i;
			std::memcpy(&i, p, 4);
			return float(i);
		}
		default:
			return NAN;
	}
}

// t's elements in ggml order, through its strides.
std::vector<float> to_float(const ggml_tensor *t) {
	std::vector<float> v;
	v.reserve(size_t(ggml_nelements(t)));
	for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
		for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
			for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
				for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
					v.push_back(elem(t, size_t(i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3])));
				}
			}
		}
	}
	return v;
}

void fill(ggml_tensor *t, std::mt19937 &rng, float lo, float hi) {
	std::uniform_real_distribution<float> u(lo, hi);
	const int64_t n = ggml_nelements(t);
	for (int64_t i = 0; i < n; ++i) {
		const float x = u(rng);
		uint8_t *p = static_cast<uint8_t *>(t->data);
		switch (t->type) {
			case GGML_TYPE_F32:
				std::memcpy(p + 4 * i, &x, 4);
				break;
			case GGML_TYPE_F16: {
				const ggml_fp16_t h = ggml_fp32_to_fp16(x);
				std::memcpy(p + 2 * i, &h, 2);
			} break;
			case GGML_TYPE_BF16: {
				const ggml_bf16_t h = ggml_fp32_to_bf16(x);
				std::memcpy(p + 2 * i, &h, 2);
			} break;
			case GGML_TYPE_I32: {
				const int32_t v = int32_t(std::floor(x));
				std::memcpy(p + 4 * i, &v, 4);
			} break;
			default:
				break;
		}
	}
}

// test-backend-ops: mse(a, b) / mse(a, 0), a the backend under test.
double nmse(const std::vector<float> &a, const std::vector<float> &b) {
	double ab = 0.0, a0 = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		ab += double(a[i] - b[i]) * double(a[i] - b[i]);
		a0 += double(a[i]) * double(a[i]);
	}
	return a0 > 0.0 ? ab / a0 : ab;
}

std::string shape(const ggml_tensor *t) {
	char b[96];
	std::snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
			(long long)t->ne[3]);
	return b;
}

enum Control { NONE, SWAP_NB };

struct Outcome {
	bool ok = false;
	bool swap_noop = true; // the control changed no address in any dispatch
	std::string line;
};

Outcome run_case(const L2Case &c, Control control) {
	Outcome o;
	const size_t mem_size = size_t(256) << 20;
	ggml_init_params ip = { mem_size, nullptr, false };
	ggml_context *ctx = ggml_init(ip);
	ggml_tensor *out = c.build(ctx);
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, 256, false);
	ggml_build_forward_expand(gf, out);

	std::mt19937 rng(1234567u);
	for (ggml_tensor *t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
		if (t->op == GGML_OP_NONE && t->view_src == nullptr && t->data != nullptr) {
			fill(t, rng, c.lo, c.hi);
		}
	}
	Mem mem{ static_cast<uint8_t *>(ggml_get_mem_buffer(ctx)), mem_size };
	const size_t used = ggml_used_mem(ctx);
	std::vector<uint8_t> snap(mem.base, mem.base + used);

	ggml_graph_compute_with_ctx(ctx, gf, 1); // the reference, ggml-cpu, one thread
	const std::vector<float> ref = to_float(out);
	std::memcpy(mem.base, snap.data(), used); // the leaves again; outputs are rewritten

	int dispatches = 0;
	std::string why;
	for (int i = 0; i < ggml_graph_n_nodes(gf) && why.empty(); ++i) {
		ggml_tensor *node = ggml_graph_node(gf, i);
		if (is_layout_only(node->op)) {
			continue;
		}
		const OpEntry *e = find_op(node);
		if (e == nullptr) {
			why = std::string("no packer for ") + ggml_op_desc(node);
			break;
		}
		uint32_t w[kWordsPerSlot];
		if (!fill_standard(w, node, host_offset, &mem, &why)) {
			break;
		}
		Pack p;
		p.node = node;
		p.w = w;
		if (!e->pack(p) || p.kernel < 0) {
			why = std::string("packer ") + e->name + " refused";
			break;
		}
		w[W_KERNEL] = uint32_t(p.kernel);
		if (control == SWAP_NB) {
			const uint32_t ws = c.control_src == 1 ? W_SRC1 : W_SRC0;
			std::swap(w[ws + T_NB + 1], w[ws + T_NB + 2]);
			// A no-op when the two strides are equal or neither dimension
			// is ever indexed past 0.
			const bool indexed = w[ws + T_NE + 1] > 1 || w[ws + T_NE + 2] > 1;
			if (w[ws + T_NB + 1] != w[ws + T_NB + 2] && indexed) {
				o.swap_noop = false;
			}
		}
		if (!run_kernel(p.kernel, w, mem.base, mem.bytes, p.groups)) {
			why = "no host runner for kernel " + std::to_string(p.kernel);
			break;
		}
		++dispatches;
	}
	const std::vector<float> emu = to_float(out);
	int64_t exact = 0;
	bool finite_match = true;
	for (size_t i = 0; i < ref.size(); ++i) {
		exact += std::memcmp(&emu[i], &ref[i], 4) == 0;
		if (std::isnan(emu[i]) != std::isnan(ref[i])) {
			finite_match = false;
		}
	}
	const double err = nmse(emu, ref);
	o.ok = why.empty() && dispatches > 0 && finite_match && err <= c.max_nmse;
	char b[512];
	std::snprintf(b, sizeof b, "%-44s out=%-18s dispatches=%d nmse=%.3e (max %.0e) exact=%lld/%zu%s%s", c.name.c_str(),
			shape(out).c_str(), dispatches, err, c.max_nmse, (long long)exact, ref.size(), why.empty() ? "" : " error: ",
			why.c_str());
	o.line = b;
	ggml_free(ctx);
	return o;
}

} // namespace

int main(int argc, char **argv) {
	Control control = NONE;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--control=swap-nb") == 0) {
			control = SWAP_NB;
		}
	}
	std::vector<L2Case> cases;
	for (L2Maker m : makers()) {
		m(cases);
	}
	std::printf("ggml-rd L2: %zu cases, %zu packers, %u kernels, control=%s\n", cases.size(), ops().size(), kKernelCount,
			control == SWAP_NB ? "swap-nb" : "none");
	int passed = 0, failed = 0, detected = 0, noop = 0;
	for (const L2Case &c : cases) {
		const Outcome o = run_case(c, control);
		if (control == NONE) {
			std::printf("%s %s\n", o.ok ? "OK  " : "FAIL", o.line.c_str());
			(o.ok ? passed : failed)++;
		} else if (o.swap_noop) {
			std::printf("NOOP %s\n", o.line.c_str());
			++noop;
			(o.ok ? passed : failed)++;
		} else {
			// The control must be caught: the case has to FAIL.
			std::printf("%s %s\n", o.ok ? "MISSED  " : "DETECTED", o.line.c_str());
			(o.ok ? failed : detected)++;
		}
	}
	bool pass;
	if (control == NONE) {
		std::printf("summary: %d/%zu cases OK\n", passed, cases.size());
		pass = failed == 0 && passed > 0;
	} else {
		std::printf("summary: swapped strides detected in %d cases, missed in %d, no-op in %d (of those, %d still OK)\n",
				detected, failed, noop, passed);
		pass = failed == 0 && detected > 0;
	}
	std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}
