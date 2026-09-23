#include "probes.h"

#include "census.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "pump/pump.h"
#include "rd_compute.h"
#include "rd_pack.h"

#include "ggml_controls.inc" // ctl_add_f32_rosrc (kernels/ggml/gen.sh, build dir)
#include "ggml_kernels.inc" // add_f32

namespace probes {

// probes_rows.cpp: NORM/RMS_NORM/SOFT_MAX GPU time on the census's shapes.
bool rows_perf(ggml_backend_t be, const std::string &arg);
// probe_conv.cpp (family K7): the census' IM2COL and CONV_3D shapes, timed.
bool conv_perf(ggml_backend_t be, const std::string &which);
// probe_graph.cpp (G3.graph, G3.cost): the apps' graphs, RD vs the in-guest CPU.
bool graph_probe(const std::string &arg);
bool cost_probe(const std::string &arg);

namespace {

struct Args {
	std::string name, arg;
};
Args g_args;
rdc::Device *g_dev = nullptr;

ggml_backend_t rd_backend() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		std::printf("ggml-rd: no RD device\n");
		return nullptr;
	}
	return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

ggml_context *make_ctx(int tensors) {
	ggml_init_params ip = {
		/* .mem_size = */ ggml_tensor_overhead() * size_t(tensors + 8) + ggml_graph_overhead_custom(size_t(tensors + 8), false),
		/* .mem_base = */ nullptr,
		/* .no_alloc = */ true,
	};
	return ggml_init(ip);
}

void result(bool ok, const char *probe) {
	std::printf("RESULT: %s (%s)\n", ok ? "PASS" : "FAIL", probe);
	std::fflush(stdout);
}

void chain(int n) {
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "chain");
		return;
	}
	const int64_t len = 256;
	ggml_context *ctx = make_ctx(n + 4);
	ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_tensor *one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_set_name(x, "x");
	ggml_set_name(one, "one");
	ggml_tensor *cur = x;
	for (int i = 0; i < n; ++i) {
		cur = ggml_add_inplace(ctx, cur, one);
	}
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, size_t(n + 8), false);
	ggml_build_forward_expand(gf, cur);
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	std::vector<float> zeros(len, 0.0f), ones(len, 1.0f), got(len, -1.0f);
	ggml_backend_tensor_set(x, zeros.data(), 0, len * 4);
	ggml_backend_tensor_set(one, ones.data(), 0, len * 4);
	const ggml_status st = ggml_backend_graph_compute(be, gf); // submit + WAIT_GPU + sync
	int64_t dispatches = 0, barriers = 0;
	ggml_backend_rd_last_graph(&dispatches, &barriers);
	ggml_backend_tensor_get(x, got.data(), 0, len * 4);
	int exact = 0;
	for (float v : got) {
		exact += v == float(n);
	}
	std::printf("PROBE chain n=%d status=%d nodes=%d dispatches=%lld barriers=%lld exact=%d/%lld x[0]=%g x[%lld]=%g "
				"barrier_all=%s\n",
			n, int(st), ggml_graph_n_nodes(gf), (long long)dispatches, (long long)barriers, exact, (long long)len,
			double(got[0]), (long long)(len - 1), double(got[len - 1]),
			std::getenv("GGML_RD_BARRIER_ALL") ? std::getenv("GGML_RD_BARRIER_ALL") : "0");
	const bool ok = st == GGML_STATUS_SUCCESS && exact == len && dispatches == n && barriers == n - 1;
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	ggml_backend_free(be);
	result(ok, "chain");
}

void independent(int n) {
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "independent");
		return;
	}
	const int64_t len = 1000;
	ggml_context *ctx = make_ctx(2 * n + 8);
	ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	std::vector<ggml_tensor *> outs;
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, size_t(2 * n + 8), false);
	for (int i = 0; i < n; ++i) {
		ggml_tensor *o = (i % 2 == 0) ? ggml_add(ctx, a, b) : ggml_mul(ctx, a, b);
		outs.push_back(o);
		ggml_build_forward_expand(gf, o);
	}
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	std::vector<float> av(len), bv(len);
	for (int64_t i = 0; i < len; ++i) {
		av[size_t(i)] = 0.25f * float(i) - 3.0f;
		bv[size_t(i)] = 1.5f - 0.125f * float(i % 17);
	}
	ggml_backend_tensor_set(a, av.data(), 0, len * 4);
	ggml_backend_tensor_set(b, bv.data(), 0, len * 4);
	const ggml_status st = ggml_backend_graph_compute(be, gf);
	int64_t dispatches = 0, barriers = 0;
	ggml_backend_rd_last_graph(&dispatches, &barriers);
	int64_t exact = 0;
	std::vector<float> got(len);
	for (int i = 0; i < n; ++i) {
		ggml_backend_tensor_get(outs[size_t(i)], got.data(), 0, len * 4);
		for (int64_t k = 0; k < len; ++k) {
			const float want = (i % 2 == 0) ? av[size_t(k)] + bv[size_t(k)] : av[size_t(k)] * bv[size_t(k)];
			exact += std::memcmp(&got[size_t(k)], &want, 4) == 0;
		}
	}
	const char *ba = std::getenv("GGML_RD_BARRIER_ALL");
	const bool all = ba != nullptr && std::atoi(ba) != 0;
	std::printf("PROBE independent n=%d status=%d dispatches=%lld barriers=%lld (want %d) exact=%lld/%lld barrier_all=%s\n", n,
			int(st), (long long)dispatches, (long long)barriers, all ? n - 1 : 0, (long long)exact,
			(long long)(len * n), all ? "1" : "0");
	const bool ok = st == GGML_STATUS_SUCCESS && exact == len * n && dispatches == n && barriers == (all ? n - 1 : 0);
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	ggml_backend_free(be);
	result(ok, "independent");
}

void files(const std::string &path) {
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "files");
		return;
	}
	// READ: the whole file, through the host.
	std::vector<uint8_t> bytes = pump::read(path, 0, 0);
	const int64_t len = int64_t(bytes.size() / 4);
	if (len == 0) {
		std::printf("PROBE files: READ %s returned %zu bytes\n", path.c_str(), bytes.size());
		ggml_backend_free(be);
		result(false, "files");
		return;
	}
	ggml_context *ctx = make_ctx(8);
	ggml_tensor *pad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 77); // x is not at offset 0
	ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_tensor *y = ggml_add(ctx, x, x);
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16, false);
	ggml_build_forward_expand(gf, y);
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	(void)pad;
	// UPLOAD: the same bytes, host file -> RD buffer, never through the guest.
	const size_t off = ggml_backend_rd_tensor_offset(x);
	const bool uploaded = ggml_backend_rd_tensor_upload(x, 0, path, 0, size_t(len) * 4);
	std::vector<float> got(static_cast<size_t>(len)), sum(static_cast<size_t>(len));
	ggml_backend_tensor_get(x, got.data(), 0, size_t(len) * 4);
	const ggml_status st = ggml_backend_graph_compute(be, gf);
	ggml_backend_tensor_get(y, sum.data(), 0, size_t(len) * 4);
	int64_t up_exact = 0, sum_exact = 0;
	for (int64_t i = 0; i < len; ++i) {
		float r;
		std::memcpy(&r, bytes.data() + 4 * i, 4);
		up_exact += std::memcmp(&got[size_t(i)], &r, 4) == 0;
		const float w = r + r;
		sum_exact += std::memcmp(&sum[size_t(i)], &w, 4) == 0;
	}
	std::printf("PROBE files path=%s read_bytes=%zu upload_offset=%zu upload_exact=%lld/%lld x+x_exact=%lld/%lld status=%d\n",
			path.c_str(), bytes.size(), off, (long long)up_exact, (long long)len, (long long)sum_exact, (long long)len,
			int(st));
	const bool ok = uploaded && up_exact == len && sum_exact == len && st == GGML_STATUS_SUCCESS;
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	ggml_backend_free(be);
	result(ok, "files");
}

// FLASH_ATTN_EXT at a census shape on the GPU: Q [128, lq, 12], K and V
// [128, lk, 12] f32, no mask, scale 1/sqrt(128). One graph of one FA node is
// computed `reps` times: every compute is one submit, a WAIT_GPU, and the
// sync on the next pump, so after the first (pipeline creation) the host's
// period between WAIT_GPUs is the node's GPU time plus one frame's overhead
// (the guest clock is not a clock, AGENTS.md: the host times it). Then 8
// sampled query rows of the output, every head, are checked against a double
// precision reference in the guest.
void fa_perf(const std::string &arg) {
	int lq = 0, lk = 0, reps = 0;
	if (std::sscanf(arg.c_str(), "%d,%d,%d", &lq, &lk, &reps) != 3 || lq < 1 || lk < 1 || reps < 1) {
		std::printf("PROBE fa_perf: bad argument '%s' (lq,lk,reps)\n", arg.c_str());
		result(false, "fa_perf");
		return;
	}
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "fa_perf");
		return;
	}
	const int d = 128, nh = 12;
	ggml_context *ctx = make_ctx(4);
	ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, lq, nh);
	ggml_tensor *k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, lk, nh);
	ggml_tensor *v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, lk, nh);
	const float scale = 1.0f / std::sqrt(float(d));
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8, false);
	ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
	ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
	ggml_build_forward_expand(gf, out);
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	auto fill = [](std::vector<float> &x, uint32_t seed) {
		uint32_t s = seed;
		for (float &e : x) {
			s = s * 1664525u + 1013904223u;
			e = float(s >> 8) * (2.0f / 16777216.0f) - 1.0f; // [-1, 1)
		}
	};
	std::vector<float> qv(size_t(d) * lq * nh), kv(size_t(d) * lk * nh), vv(size_t(d) * lk * nh);
	fill(qv, 1u);
	fill(kv, 2u);
	fill(vv, 3u);
	ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * 4);
	ggml_backend_tensor_set(k, kv.data(), 0, kv.size() * 4);
	ggml_backend_tensor_set(v, vv.data(), 0, vv.size() * 4);
	ggml_status st = GGML_STATUS_SUCCESS;
	int64_t dispatches = 0, barriers = 0;
	for (int r = 0; r < reps && st == GGML_STATUS_SUCCESS; ++r) {
		st = ggml_backend_graph_compute(be, gf);
		int64_t dd = 0;
		ggml_backend_rd_last_graph(&dd, &barriers);
		dispatches += dd;
	}
	// 8 sampled rows of the output, every head, vs a double reference.
	double max_err = 0.0;
	std::vector<float> row(size_t(d) * nh);
	std::vector<double> sc(static_cast<size_t>(lk)), acc(static_cast<size_t>(d));
	for (int si = 0; si < 8; ++si) {
		const int iq = int((int64_t(si) * (lq - 1)) / 7);
		ggml_backend_tensor_get(out, row.data(), size_t(iq) * d * nh * 4, row.size() * 4); // dst [d, nh, lq]
		for (int h = 0; h < nh; ++h) {
			const float *qr = &qv[(size_t(h) * lq + iq) * d];
			double m = -INFINITY;
			for (int j = 0; j < lk; ++j) {
				const float *kr = &kv[(size_t(h) * lk + j) * d];
				double s = 0.0;
				for (int c = 0; c < d; ++c) {
					s += double(qr[c]) * double(kr[c]);
				}
				sc[size_t(j)] = s * double(scale);
				m = std::max(m, sc[size_t(j)]);
			}
			double l = 0.0;
			std::fill(acc.begin(), acc.end(), 0.0);
			for (int j = 0; j < lk; ++j) {
				const double p = std::exp(sc[size_t(j)] - m);
				l += p;
				const float *vr = &vv[(size_t(h) * lk + j) * d];
				for (int c = 0; c < d; ++c) {
					acc[size_t(c)] += p * double(vr[c]);
				}
			}
			for (int c = 0; c < d; ++c) {
				max_err = std::max(max_err, std::fabs(double(row[size_t(h) * d + c]) - acc[size_t(c)] / l));
			}
		}
	}
	const double gflop = 4.0 * d * double(lq) * double(lk) * nh * 1e-9; // QK^T and PV, per node
	std::printf("PROBE fa_perf lq=%d lk=%d d=%d heads=%d reps=%d status=%d dispatches=%lld barriers=%lld "
				"gflop_per_node=%.2f sampled_max_abs_err=%.3e serial=%s\n",
			lq, lk, d, nh, reps, int(st), (long long)dispatches, (long long)barriers, gflop, max_err,
			std::getenv("GGML_RD_SERIAL") ? std::getenv("GGML_RD_SERIAL") : "0");
	const bool ok = st == GGML_STATUS_SUCCESS && dispatches == reps && barriers == 0 && max_err < 1e-4;
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	ggml_backend_free(be);
	result(ok, "fa_perf");
}

bool rd_offset(const ggml_tensor *t, void *, uint64_t *off) {
	*off = ggml_backend_rd_tensor_offset(t);
	return true;
}

void alias(bool rw) {
	const char *probe = rw ? "alias rw" : "alias ro (control)";
	ggml_backend_t be = rd_backend();
	if (be == nullptr || g_dev == nullptr) {
		result(false, probe);
		return;
	}
	const int rounds = 1000;
	const int64_t len = 4096;
	ggml_context *ctx = make_ctx(8);
	ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_tensor *one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
	ggml_tensor *node = ggml_add_inplace(ctx, x, one); // a view of x: b1 and b4 are x
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	std::vector<float> zeros(size_t(len), 0.0f), ones(size_t(len), 1.0f), got(size_t(len), -1.0f);
	ggml_backend_tensor_set(x, zeros.data(), 0, size_t(len) * 4);
	ggml_backend_tensor_set(one, ones.data(), 0, size_t(len) * 4);
	ggml_backend_rd_ensure_idle();

	// The words of every round, from the backend's own packer.
	std::string why = "no packer";
	uint32_t w[ggml_rd::kWordsPerSlot];
	ggml_rd::Pack p;
	p.node = node;
	p.w = w;
	const ggml_rd::OpEntry *e = ggml_rd::find_op(node);
	bool ok = e != nullptr && ggml_rd::fill_standard(w, node, rd_offset, nullptr, &why) && e->pack(p) && p.kernel >= 0;
	w[ggml_rd::W_KERNEL] = uint32_t(p.kernel);

	rdc::Device &d = *g_dev;
	const char *kname = rw ? "add_f32" : "ctl_add_f32_rosrc";
	const uint8_t *spv = nullptr;
	size_t spv_bytes = 0;
	if (rw) {
		const ggml_kernels::Entry *k = ggml_kernels::find(kname);
		spv = k ? k->bytes : nullptr;
		spv_bytes = k ? k->size : 0;
	} else {
		const ggml_controls::Entry *k = ggml_controls::find(kname);
		spv = k ? k->bytes : nullptr;
		spv_bytes = k ? k->size : 0;
	}
	::RID shader, pipeline, params, ubo, set0, set1;
	if (ok && spv != nullptr) {
		shader = d.shader_from_spirv(spv, spv_bytes);
		pipeline = shader.index ? d.compute_pipeline(shader) : ::RID();
		params = d.storage_buffer(ggml_rd::kSlotBytes, w);
		const uint32_t slot[4] = { 0, 0, 0, 0 };
		ubo = d.uniform_buffer(sizeof slot, slot);
		const ::RID xb = ggml_backend_rd_buffer_rid(buf);
		const int SB = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
		set0 = pipeline.index ? d.uniform_set(shader, { { 0, SB, params }, { 1, SB, xb }, { 2, SB, xb }, { 3, SB, xb }, { 4, SB, xb } }, 0) : ::RID();
		set1 = pipeline.index ? d.uniform_set(shader, { { 0, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, ubo } }, 1) : ::RID();
		ok = set0.index != 0 && set1.index != 0;
		why = ok ? "" : d.error();
	}
	if (ok) {
		d.list_begin();
		d.bind_pipeline(pipeline);
		d.bind_uniform_set(set0, 0);
		d.bind_uniform_set(set1, 1);
		for (int i = 0; i < rounds; ++i) {
			if (i > 0) {
				d.barrier();
			}
			d.dispatch(p.groups[0], p.groups[1], p.groups[2]);
		}
		d.list_end();
		d.submit();
		pump::wait_gpu(); // the sync lands a frame later (rule 4)
		d.sync();
		ggml_backend_tensor_get(x, got.data(), 0, size_t(len) * 4);
	}
	int64_t exact = 0;
	float lo = 1e30f, hi = -1e30f;
	double sum = 0.0;
	for (float v : got) {
		exact += v == float(rounds);
		lo = v < lo ? v : lo;
		hi = v > hi ? v : hi;
		sum += v;
	}
	std::printf("PROBE alias %s kernel=%s rounds=%d barriers=%d elements=%lld exact=%lld/%lld min=%g max=%g mean=%.1f "
				"(want %d)%s%s\n",
			rw ? "rw" : "ro", kname, rounds, rounds - 1, (long long)len, (long long)exact, (long long)len, double(lo),
			double(hi), sum / double(len), rounds, why.empty() ? "" : " error: ", why.c_str());
	for (::RID r : { set0, set1, ubo, params, pipeline, shader }) {
		d.free_rid(r);
	}
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	ggml_backend_free(be);
	if (!ok) {
		result(false, probe);
	} else if (rw) {
		result(exact == len, probe);
	} else {
		result(exact < len, probe); // the control must show the hazard
	}
}

// --- perf <set>: GPU time per op on the census's hottest shapes -------------
//
// For each case: one graph of K copies of the op (each into its own output,
// K sized to ~512 MiB of outputs), timed on the GPU by ggml-rd's timestamps
// around the compute list, and the same with one copy; per op =
// (T_K - T_1) / (K - 1), which cancels the list's fixed cost. Under
// GGML_RD_BARRIER_ALL=1 every dispatch is followed by a barrier: the
// serialised latency a dependent chain pays. Sources hold zeros (buffers are
// cleared) except GET_ROWS' indices, which spread over the rows.
struct PerfCase {
	const char *name;
	ggml_type type;
	int64_t ne[4]; // src0
	int64_t ne1[4]; // src1 (CONCAT), the repeat target, GET_ROWS' row count
	int kind; // 0 concat(dim = arg), 1 cpy, 2 repeat, 3 cont (arg 0: permute 0,2,1,3; 1: transpose), 4 get_rows, 5 cpy to type2
	int arg;
	ggml_type type2;
};

ggml_tensor *perf_node(ggml_context *ctx, const PerfCase &pc, ggml_tensor *a, ggml_tensor *b) {
	switch (pc.kind) {
		case 0:
			return ggml_concat(ctx, a, b, pc.arg);
		case 1:
			return ggml_cpy(ctx, a, ggml_new_tensor(ctx, pc.type, 4, pc.ne));
		case 2:
			return ggml_repeat(ctx, a, ggml_new_tensor(ctx, pc.type, 4, pc.ne1));
		case 3:
			return pc.arg == 0 ? ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)) : ggml_cont(ctx, ggml_transpose(ctx, a));
		case 4:
			return ggml_get_rows(ctx, a, b);
		default:
			return ggml_cpy(ctx, a, ggml_new_tensor(ctx, pc.type2, 4, pc.ne));
	}
}

// Bytes one op reads and writes.
int64_t perf_bytes(const ggml_tensor *o) {
	int64_t n = int64_t(ggml_nbytes(o));
	const ggml_tensor *s0 = o->src[0];
	if (o->op == GGML_OP_GET_ROWS) {
		n += ggml_nelements(o) * int64_t(ggml_type_size(s0->type)) + int64_t(ggml_nbytes(o->src[1]));
	} else if (o->op == GGML_OP_CONCAT) {
		n += int64_t(ggml_nbytes(s0)) + int64_t(ggml_nbytes(o->src[1]));
	} else if (o->op == GGML_OP_REPEAT) {
		n += int64_t(ggml_nbytes(s0)); // read once per output element, from the cache
	} else {
		n += int64_t(ggml_nbytes(s0));
	}
	return n;
}

// GPU ns of one graph of k copies (-1 on failure).
int64_t perf_graph(ggml_backend_t be, const PerfCase &pc, int k, int64_t *barriers, int64_t *bytes) {
	ggml_context *ctx = make_ctx(3 * k + 8);
	ggml_tensor *a = ggml_new_tensor(ctx, pc.type, 4, pc.ne);
	ggml_tensor *b = nullptr;
	if (pc.kind == 0) {
		b = ggml_new_tensor(ctx, pc.type, 4, pc.ne1);
	} else if (pc.kind == 4) {
		b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pc.ne1[0]);
	}
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, size_t(3 * k + 8), false);
	ggml_tensor *o = nullptr;
	for (int i = 0; i < k; ++i) {
		o = perf_node(ctx, pc, a, b);
		ggml_build_forward_expand(gf, o);
	}
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	if (buf == nullptr) {
		ggml_free(ctx);
		return -1;
	}
	if (pc.kind == 4) {
		std::vector<int32_t> rows(size_t(pc.ne1[0]));
		for (size_t i = 0; i < rows.size(); ++i) {
			rows[i] = int32_t((i * 7919) % size_t(pc.ne[1]));
		}
		ggml_backend_tensor_set(b, rows.data(), 0, rows.size() * 4);
	}
	*bytes = perf_bytes(o);
	const ggml_status st = ggml_backend_graph_compute(be, gf);
	int64_t dispatches = 0;
	ggml_backend_rd_last_graph(&dispatches, barriers);
	const int64_t ns = st == GGML_STATUS_SUCCESS && dispatches == k ? ggml_backend_rd_last_gpu_ns() : -1;
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	return ns;
}

void perf(const std::string &set) {
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "perf");
		return;
	}
	const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16, BF16 = GGML_TYPE_BF16;
	// The census's hottest data-movement shapes (skin-tokens unless noted).
	const PerfCase move_cases[] = {
		{ "CONCAT f32 [128,8,514]+[128,8,1] dim2 (KV cache)", F32, { 128, 8, 514, 1 }, { 128, 8, 1, 1 }, 0, 2, F32 },
		{ "CONCAT f32 [128,8,515,1]+[128,8,515,1] dim3", F32, { 128, 8, 515, 1 }, { 128, 8, 515, 1 }, 0, 3, F32 },
		{ "CPY f32 [128,8]", F32, { 128, 8, 1, 1 }, { 0, 0, 0, 0 }, 1, 0, F32 },
		{ "REPEAT f32 [128,8,1,514]->[128,8,2,514]", F32, { 128, 8, 1, 514 }, { 128, 8, 2, 514 }, 2, 0, F32 },
		{ "CONT f32 permute(0,2,1,3) [128,8,2,514]", F32, { 128, 8, 2, 514 }, { 0, 0, 0, 0 }, 3, 0, F32 },
		{ "CONT f32 transpose [128,515,16,2]", F32, { 128, 515, 16, 2 }, { 0, 0, 0, 0 }, 3, 1, F32 },
		{ "GET_ROWS f32 [896,33036] x10", F32, { 896, 33036, 1, 1 }, { 10, 0, 0, 0 }, 4, 0, F32 },
		{ "GET_ROWS f32 [512,246] x246 (pixal3d)", F32, { 512, 246, 1, 1 }, { 246, 0, 0, 0 }, 4, 0, F32 },
		{ "CONT f16 permute(0,2,1,3) [1024,1024] (pixal3d)", F16, { 1024, 1024, 1, 1 }, { 0, 0, 0, 0 }, 3, 0, F16 },
		{ "CONCAT f32 [1,64,12,4096]x2 dim0 (pixal3d)", F32, { 1, 64, 12, 4096 }, { 1, 64, 12, 4096 }, 0, 0, F32 },
		{ "CPY bf16->f32 [1536,4096]", BF16, { 1536, 4096, 1, 1 }, { 0, 0, 0, 0 }, 5, 0, F32 },
		{ "CPY f32->f16 [1536,4096]", F32, { 1536, 4096, 1, 1 }, { 0, 0, 0, 0 }, 5, 0, F16 },
	};
	ggml_backend_rd_set_timestamps(true);
	bool ok = true;
	for (const PerfCase &pc : move_cases) {
		int64_t bar1 = 0, bark = 0, bytes = 0;
		perf_graph(be, pc, 1, &bar1, &bytes); // warm: pipelines, sets
		const int64_t t1 = perf_graph(be, pc, 1, &bar1, &bytes);
		const int64_t out_bytes = std::max<int64_t>(bytes / 2, 1);
		const int k = int(std::min<int64_t>(256, std::max<int64_t>(16, (int64_t(512) << 20) / out_bytes)));
		const int64_t tk = perf_graph(be, pc, k, &bark, &bytes);
		const bool good = t1 > 0 && tk > 0;
		ok = ok && good;
		const double per_ns = good ? double(tk - t1) / double(k - 1) : -1.0;
		std::printf("PROBE perf %s: k=%d barriers=%lld t1_us=%.1f tk_us=%.1f per_op_us=%.2f bytes=%lld GBps=%.0f\n",
				pc.name, k, (long long)bark, double(t1) / 1e3, double(tk) / 1e3, per_ns / 1e3, (long long)bytes,
				per_ns > 0 ? double(bytes) / per_ns : 0.0);
		std::fflush(stdout);
	}
	ggml_backend_rd_set_timestamps(false);
	ggml_backend_free(be);
	(void)set;
	result(ok, "perf");
}

// --- mm_perf: MUL_MAT timing on the census's hottest shapes ------------------

// The host's clock (Time.get_ticks_usec(), a host call): the guest clock
// jumps between time bases (AGENTS.md facts), so every time here is the
// host's.
double host_us() {
	Object t("Time");
	return double(int64_t(t.call("get_ticks_usec")));
}

struct MmShape {
	const char *label;
	ggml_type ta, tb;
	int64_t m, n, k;
	int64_t ne02, ne03, ne12, ne13;
	bool kv; // src0 is a K cache [K, ne02, M, ne03] viewed as [K, M, ne02, ne03] (the census's RC)
	int r1, r2; // mul_mats in the short and the long graph
	const char *census;
};

const MmShape kMmShapes[] = {
	{ "gflops_f16", GGML_TYPE_F16, GGML_TYPE_F32, 4096, 1024, 1536, 1, 1, 1, 1, false, 1, 4, "the 4096x1536x1024 benchmark" },
	{ "gflops_bf16", GGML_TYPE_BF16, GGML_TYPE_F32, 4096, 1024, 1536, 1, 1, 1, 1, false, 1, 4, "the 4096x1536x1024 benchmark" },
	{ "gflops_f32", GGML_TYPE_F32, GGML_TYPE_F32, 4096, 1024, 1536, 1, 1, 1, 1, false, 1, 4, "the same in f32" },
	{ "st_decode_2048x896", GGML_TYPE_F16, GGML_TYPE_F32, 2048, 1, 896, 1, 1, 2, 1, false, 8, 72, "skin-tokens vec, 32172 calls" },
	{ "st_decode_1024x896", GGML_TYPE_F16, GGML_TYPE_F32, 1024, 1, 896, 1, 1, 2, 1, false, 8, 72, "skin-tokens vec, 21448 calls" },
	{ "st_decode_896x2048", GGML_TYPE_F16, GGML_TYPE_F32, 896, 1, 2048, 1, 1, 2, 1, false, 8, 72, "skin-tokens vec, 21448 calls" },
	{ "st_attn_kq_rc", GGML_TYPE_F32, GGML_TYPE_F32, 515, 1, 128, 16, 2, 16, 2, true, 8, 72, "skin-tokens RC vec, 10724 calls" },
	{ "st_mat_1024x54000x512", GGML_TYPE_F16, GGML_TYPE_F32, 1024, 54000, 512, 1, 1, 1, 1, false, 1, 3, "skin-tokens mat, 3183 calls" },
	{ "p3_bf16_4608x4096x1536", GGML_TYPE_BF16, GGML_TYPE_F32, 4608, 4096, 1536, 1, 1, 1, 1, false, 1, 3, "Pixal3D bf16, 120 calls" },
	{ "p3_f16_1024x1029x1024", GGML_TYPE_F16, GGML_TYPE_F32, 1024, 1029, 1024, 1, 1, 1, 1, false, 1, 5, "Pixal3D f16, 663 calls" },
	{ "p3_f32_attn_1029x1029x64x16", GGML_TYPE_F32, GGML_TYPE_F32, 1029, 1029, 64, 16, 1, 16, 1, false, 1, 3, "Pixal3D f32 batch, 24 calls" },
};

uint32_t g_lcg = 12345u;
float rnd() {
	g_lcg = g_lcg * 1664525u + 1013904223u;
	return float(g_lcg >> 8) * (2.0f / 16777216.0f) - 1.0f; // [-1, 1)
}

// n values of type t in [-1, 1), as bytes.
std::vector<uint8_t> random_bytes(ggml_type t, int64_t n) {
	std::vector<uint8_t> v(size_t(n) * ggml_type_size(t));
	for (int64_t i = 0; i < n; ++i) {
		const float x = rnd();
		if (t == GGML_TYPE_F32) {
			std::memcpy(v.data() + 4 * i, &x, 4);
		} else if (t == GGML_TYPE_F16) {
			const ggml_fp16_t h = ggml_fp32_to_fp16(x);
			std::memcpy(v.data() + 2 * i, &h, 2);
		} else {
			const ggml_bf16_t h = ggml_fp32_to_bf16(x);
			std::memcpy(v.data() + 2 * i, &h, 2);
		}
		if ((i & 0xFFFFF) == 0xFFFFF) {
			pump::coop();
		}
	}
	return v;
}

double elem_at(const std::vector<uint8_t> &bytes, ggml_type t, size_t off) {
	if (t == GGML_TYPE_F32) {
		float f;
		std::memcpy(&f, bytes.data() + off, 4);
		return f;
	}
	uint16_t h;
	std::memcpy(&h, bytes.data() + off, 2);
	if (t == GGML_TYPE_F16) {
		return ggml_fp16_to_fp32(h);
	}
	ggml_bf16_t b;
	b.bits = h;
	return ggml_bf16_to_fp32(b);
}

// Host microseconds for one graph: compute (pack, record, submit) and the
// synchronize that waits for it (a frame later, rule 4).
double time_graph(ggml_backend_t be, ggml_cgraph *g, ggml_status *st) {
	const double t0 = host_us();
	*st = ggml_backend_graph_compute(be, g);
	ggml_backend_synchronize(be);
	return host_us() - t0;
}

bool mm_shape(ggml_backend_t be, const MmShape &s) {
	const int trials = 3;
	ggml_context *ctx = make_ctx(s.r2 + 8);
	ggml_tensor *abase = s.kv ? ggml_new_tensor_4d(ctx, s.ta, s.k, s.ne02, s.m, s.ne03)
							  : ggml_new_tensor_4d(ctx, s.ta, s.k, s.m, s.ne02, s.ne03);
	ggml_tensor *a = s.kv ? ggml_permute(ctx, abase, 0, 2, 1, 3) : abase;
	ggml_tensor *b = ggml_new_tensor_4d(ctx, s.tb, s.k, s.n, s.ne12, s.ne13);
	std::vector<ggml_tensor *> outs;
	ggml_cgraph *g1 = ggml_new_graph_custom(ctx, size_t(s.r2 + 8), false);
	ggml_cgraph *g2 = ggml_new_graph_custom(ctx, size_t(s.r2 + 8), false);
	for (int i = 0; i < s.r2; ++i) {
		outs.push_back(ggml_mul_mat(ctx, a, b));
		if (i < s.r1) {
			ggml_build_forward_expand(g1, outs.back());
		}
		ggml_build_forward_expand(g2, outs.back());
	}
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
	if (buf == nullptr) {
		std::printf("PROBE mm_perf %s: could not allocate %.0f MB\n", s.label,
				double(ggml_nbytes(abase) + ggml_nbytes(b) + s.r2 * ggml_nbytes(outs[0])) / 1048576.0);
		ggml_free(ctx);
		return false;
	}
	const std::vector<uint8_t> ab = random_bytes(s.ta, ggml_nelements(abase));
	const std::vector<uint8_t> bb = random_bytes(s.tb, ggml_nelements(b));
	ggml_backend_tensor_set(abase, ab.data(), 0, ab.size());
	ggml_backend_tensor_set(b, bb.data(), 0, bb.size());

	ggml_status st = GGML_STATUS_SUCCESS, st1 = GGML_STATUS_SUCCESS;
	time_graph(be, g2, &st); // warm-up: pipelines and sets
	int64_t dispatches = 0, barriers = 0;
	ggml_backend_rd_last_graph(&dispatches, &barriers);
	double t1 = 1e30, t2 = 1e30;
	for (int t = 0; t < trials; ++t) {
		const double x1 = time_graph(be, g1, &st1);
		const double x2 = time_graph(be, g2, &st);
		t1 = x1 < t1 ? x1 : t1;
		t2 = x2 < t2 ? x2 : t2;
	}
	const double per_op_us = (t2 - t1) / double(s.r2 - s.r1);
	const double flops = 2.0 * double(s.m) * double(s.n) * double(s.k) * double(s.ne12) * double(s.ne13);

	// Check the last output of the long graph against a double sum over the
	// same inputs: dst columns j in {0, N/2, N-1}, first and last batch.
	const ggml_tensor *o = outs.back();
	const int64_t r2b = s.ne12 / s.ne02, r3b = s.ne13 / s.ne03;
	std::vector<int64_t> js = { 0, s.n / 2, s.n - 1 };
	std::vector<std::pair<int64_t, int64_t>> batches = { { 0, 0 }, { s.ne12 - 1, s.ne13 - 1 } };
	std::vector<float> got(size_t(s.m));
	double se = 0.0, sr = 0.0, maxrel = 0.0;
	int64_t checked = 0;
	for (auto [b2, b3] : batches) {
		for (int64_t j : js) {
			ggml_backend_tensor_get(const_cast<ggml_tensor *>(o), got.data(), size_t(j * o->nb[1] + b2 * o->nb[2] + b3 * o->nb[3]),
					size_t(s.m) * 4);
			for (int64_t i = 0; i < s.m; ++i) {
				double ref = 0.0;
				for (int64_t k = 0; k < s.k; ++k) {
					const size_t ao = size_t(k * a->nb[0] + i * a->nb[1] + (b2 / r2b) * a->nb[2] + (b3 / r3b) * a->nb[3]);
					const size_t bo = size_t(k * b->nb[0] + j * b->nb[1] + b2 * b->nb[2] + b3 * b->nb[3]);
					ref += elem_at(ab, s.ta, ao) * elem_at(bb, s.tb, bo);
				}
				const double d = double(got[size_t(i)]) - ref;
				se += d * d;
				sr += ref * ref;
				const double rel = std::fabs(d) / (std::fabs(ref) > 1e-3 ? std::fabs(ref) : 1e-3);
				maxrel = rel > maxrel ? rel : maxrel;
				++checked;
			}
			pump::coop();
		}
	}
	const double nmse = sr > 0 ? se / sr : se;
	const bool ok = st == GGML_STATUS_SUCCESS && st1 == GGML_STATUS_SUCCESS && nmse < 1e-8 && per_op_us > 0.0;
	std::printf("PROBE mm_perf %-28s %s x %s M=%lld N=%lld K=%lld batch=%lldx%lld (bcast %lldx%lld)%s kernel=%s "
				"per_op_us=%.1f gflops=%.1f graph_us(r=%d)=%.0f graph_us(r=%d)=%.0f dispatches=%lld barriers=%lld "
				"checked=%lld nmse=%.2e maxrel=%.2e [%s] %s\n",
			s.label, ggml_type_name(s.ta), ggml_type_name(s.tb), (long long)s.m, (long long)s.n, (long long)s.k,
			(long long)s.ne12, (long long)s.ne13, (long long)r2b, (long long)r3b, s.kv ? " src0=permuted-kv" : "",
			s.n <= 4 ? "vec" : "tiled", per_op_us, flops / per_op_us / 1e3, s.r1, t1, s.r2, t2, (long long)dispatches,
			(long long)barriers, (long long)checked, nmse, maxrel, s.census, ok ? "ok" : "BAD");
	std::fflush(stdout);
	ggml_backend_buffer_free(buf);
	ggml_free(ctx);
	return ok;
}

void mm_perf(const std::string &which) {
	ggml_backend_t be = rd_backend();
	if (be == nullptr) {
		result(false, "mm_perf");
		return;
	}
	bool ok = true;
	int ran = 0;
	for (const MmShape &s : kMmShapes) {
		if (which != "all" && which.find(s.label) == std::string::npos) {
			continue;
		}
		ok = mm_shape(be, s) && ok;
		++ran;
	}
	ggml_backend_free(be);
	result(ok && ran > 0, "mm_perf");
}

void job(void *) {
	const std::string &name = g_args.name;
	const int n = g_args.arg.empty() ? 0 : std::atoi(g_args.arg.c_str());
	if (name == "chain") {
		chain(n > 0 ? n : 256);
	} else if (name == "independent") {
		independent(n > 0 ? n : 64);
	} else if (name == "files") {
		files(g_args.arg);
	} else if (name == "alias") {
		alias(g_args.arg != "ro");
	} else if (name == "perf" && g_args.arg == "move") {
		perf(g_args.arg); // K2's data-movement set; other perf sets are census rows
	} else if (census::known(name)) {
		census::run(name, g_args.arg.empty() ? "all" : g_args.arg);
	} else if (name == "rows_perf") {
		ggml_backend_t be = rd_backend();
		result(be != nullptr && rows_perf(be, g_args.arg), "rows_perf");
		if (be != nullptr) {
			ggml_backend_free(be);
		}
	} else if (name == "mm_perf") {
		mm_perf(g_args.arg.empty() ? "all" : g_args.arg);
	} else if (name == "conv_perf") {
		ggml_backend_t be = rd_backend();
		const bool ok = be != nullptr && conv_perf(be, g_args.arg);
		if (be != nullptr) {
			ggml_backend_free(be);
		}
		result(ok, "conv_perf");
	} else if (name == "fa_perf") {
		fa_perf(g_args.arg);
	} else if (name == "graph") {
		result(graph_probe(g_args.arg), ("graph " + g_args.arg).c_str());
	} else if (name == "cost") {
		result(cost_probe(g_args.arg), ("cost " + g_args.arg).c_str());
	}
	std::printf("ggml_test: rd stats %s\n", ggml_backend_rd_stats().c_str());
	std::fflush(stdout);
}

} // namespace

void set_device(rdc::Device *dev) {
	g_dev = dev;
}

bool start(const std::string &name, const std::string &arg, std::string &err) {
	if (name != "chain" && name != "independent" && name != "files" && name != "alias" && name != "perf"
			&& name != "rows_perf" && name != "mm_perf" && name != "conv_perf"
			&& name != "fa_perf" && name != "graph" && name != "cost" && !census::known(name)) {
		err = "unknown probe '" + name + "' (chain, independent, files, alias, census, perf, rows_perf, mm_perf, conv_perf, fa_perf, graph, cost)";
		return false;
	}
	g_args = Args{ name, arg };
	if (!pump::start(&job, nullptr)) {
		err = "a job is running";
		return false;
	}
	return true;
}

} // namespace probes
