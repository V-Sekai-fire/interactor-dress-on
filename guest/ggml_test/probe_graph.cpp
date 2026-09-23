// Gate 3 G3.graph and G3.cost: the apps' own graph builders
// (app_graphs/, copied from skin-tokens-ggml and pixal3d-ggml) on random
// weights, run in the guest on ggml-rd ONLY. The references are computed on
// the HOST (tests/ggml_graph_oracle: ggml-vulkan on the GPU for the DiT block,
// host-native ggml-cpu for the small graphs), never on the in-guest ggml-cpu:
// the guest CPU (rv64gc, one thread, ~0.1 GFLOP/s) is far too slow for
// anything beyond G3.ops' single-op cases (a 4096-token DiT block's two
// reference arms would be hours). graph_nets.cpp builds the nets, the same
// file the oracle compiles.
//
//   graph <qwen|dit|sconv>[:res]   G3.graph for one graph (graph_nets.h):
//     rd native / rd f32   the weights in the model's type (f16, bf16), and
//                          the same values widened to f32; both arms' outputs
//                          are kept for the host (graph_dump_list/_chunk,
//                          main.cpp's ggml_dump_list/ggml_dump_chunk), which
//                          compares them with the oracle: rel-L2 <= 1e-3
//                          native, <= 1e-4 f32.
//   and here, on the rd native net: barrier elision vs GGML_RD_BARRIER_ALL=1
//   and a second elision run, every output bit for bit; then the control, one
//   run per barrier elision placed (at most 48, spread) with
//   GGML_RD_DROP_BARRIER=k, each compared bit for bit with the barrier-all
//   outputs: a run that differs has "detected" its dropped barrier. Before
//   every RD run the graph's compute buffers are cleared to 0, so a read
//   that races its producer sees zeros, not the previous run's result.
//
//   cost <decode|dit>   G3.cost, RD only: the host's microseconds per graph
//     and per node (Time.get_ticks_usec around graph_compute; its phases from
//     GGML_RD_PROFILE=1; a per-kernel table from one GGML_RD_PROFILE=2 run),
//     dispatches and barriers per graph, frames per graph (the pump's
//     WAIT_GPU + COOP yields from building the graph to reading its output),
//     and the GPU time (timestamps around the compute list):
//     decode  a skin-tokens decode step as the app builds it every token
//             (qwen_graph_evaluator::decode: embedding rows, 28 layers, norm,
//             logits; a new context and allocator per step), 2 beams, past 514
//     dit     a Pixal3D flow forward (30 blocks, as trellis2_ss_flow_forward
//             builds it every call), bf16 weights
//   Layer/block 0's weights are random and copied into the other layers on
//   the GPU (distinct buffers, so no layer reads another's from cache).
#include <api.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "app_graphs/app_graphs.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "graph_nets.h"
#include "pump/pump.h"
#include "rd_compute.h"

namespace probes {

namespace {

using app_graphs::WeightFn;
using namespace graph_nets;

// The last G3.graph run's RD outputs, for the host oracle: "<arm>/<output>"
// (arm rd_native or rd_f32), f32 little-endian.
std::vector<std::pair<std::string, std::vector<float>>> g_dump;

ggml_backend_t rd_backend_or_null() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		return nullptr;
	}
	return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

// The graph's own buffers (gallocr's), not the leaves'.
std::vector<ggml_backend_buffer_t> compute_buffers(const Net &n) {
	std::set<ggml_backend_buffer_t> s;
	for (int i = 0; i < ggml_graph_n_nodes(n.gf); ++i) {
		const ggml_tensor *t = ggml_graph_node(n.gf, i);
		const ggml_tensor *base = t->view_src ? t->view_src : t;
		if (base->buffer != nullptr && base->buffer != n.lbuf) {
			s.insert(base->buffer);
		}
	}
	return std::vector<ggml_backend_buffer_t>(s.begin(), s.end());
}

struct RunInfo {
	ggml_status st = GGML_STATUS_FAILED;
	int64_t dispatches = 0, barriers = 0;
	int64_t compute_us = 0; // RD: graph_compute's host time from entry to submit (GGML_RD_PROFILE=1)
	int64_t frames = 0; // WAIT_GPU + COOP yields from compute to the outputs read
	std::string dropped;
};

int64_t frame_yields() {
	return pump::yields(pump::WAIT_GPU) + pump::yields(pump::COOP);
}

RunInfo run(Net &n, Outs &out, bool clear) {
	RunInfo r;
	if (n.rd && clear) {
		for (ggml_backend_buffer_t b : compute_buffers(n)) {
			ggml_backend_buffer_clear(b, 0);
		}
		ggml_backend_rd_ensure_idle();
	}
	const int64_t f0 = frame_yields();
	if (n.rd) {
		ggml_backend_rd_set_profile(1);
	}
	r.st = ggml_backend_graph_compute(n.be, n.gf); // ggml's: compute, then synchronize
	if (n.rd) {
		ggml_backend_rd_set_profile(0);
		r.compute_us = ggml_backend_rd_last_profile().us_total; // entry to submit
		ggml_backend_rd_last_graph(&r.dispatches, &r.barriers);
		r.dropped = ggml_backend_rd_last_dropped();
	}
	out = read_outs(n);
	r.frames = frame_yields() - f0;
	return r;
}

bool all_finite(const Outs &o) {
	for (const auto &x : o) {
		if (!finite(x.second)) {
			return false;
		}
	}
	return !o.empty();
}

void set_env(const char *k, const char *v) {
	if (v) {
		setenv(k, v, 1);
	} else {
		unsetenv(k);
	}
}

// --- G3.graph -------------------------------------------------------------------------

bool graph_one(const std::string &which, int res) {
	BuildFn build;
	std::string native, resid;
	const char *g = which.c_str();
	if (!graph_builder(which, res, build, native, resid)) {
		std::printf("PROBE graph: unknown graph '%s' (qwen, dit, sconv)\n", g);
		return false;
	}
	for (const char *k : { "GGML_RD_BARRIER_ALL", "GGML_RD_DROP_BARRIER", "GGML_RD_FAULT", "GGML_RD_PROFILE" }) {
		unsetenv(k);
	}
	g_dump.clear();
	const int64_t t_start = rdc::host_usec();
	bool ok = true;

	// ggml-rd, the native arm: elision, barrier-all, elision again, the drop sweep.
	Outs rd_nat, rd_all, rd_again, rd_f32;
	RunInfo r_el, r_all, r_again, r_f32;
	int drops_run = 0, drops_detected = 0;
	int nodes = 0;
	{
		Net n;
		n.be = rd_backend_or_null();
		n.rd = true;
		if (n.be == nullptr) {
			std::printf("PROBE graph %s: no RD device\n", g);
			return false;
		}
		try {
			build(n, false);
			if (!allocate(n)) {
				throw std::runtime_error("allocation");
			}
			nodes = ggml_graph_n_nodes(n.gf);
			// A first run makes the pipelines and sets (COOP yields); not compared.
			Outs warm;
			run(n, warm, true);
			const int64_t t_el = rdc::host_usec();
			r_el = run(n, rd_nat, true);
			const int64_t t_el_end = rdc::host_usec();
			set_env("GGML_RD_BARRIER_ALL", "1");
			r_all = run(n, rd_all, true);
			set_env("GGML_RD_BARRIER_ALL", nullptr);
			r_again = run(n, rd_again, true);
			std::printf("PROBE graph %s rd_native nodes=%d dispatches=%lld barriers=%lld (barrier_all: %lld) "
						"host_us(entry..submit)=%lld frames=%lld run_ms=%.1f status=%d/%d/%d\n",
					g, nodes, (long long)r_el.dispatches, (long long)r_el.barriers, (long long)r_all.barriers,
					(long long)r_el.compute_us, (long long)r_el.frames, (t_el_end - t_el) / 1000.0, int(r_el.st),
					int(r_all.st), int(r_again.st));
			// The control: leave out one barrier elision placed.
			const int64_t nb = r_el.barriers;
			std::vector<int64_t> ks;
			const int64_t kmax = 48;
			for (int64_t i = 0; i < std::min(nb, kmax); ++i) {
				ks.push_back(nb <= kmax ? i + 1 : 1 + (i * (nb - 1)) / (kmax - 1));
			}
			for (int64_t k : ks) {
				set_env("GGML_RD_DROP_BARRIER", std::to_string(k).c_str());
				Outs d;
				const RunInfo rk = run(n, d, true);
				size_t differing = 0;
				const bool same = identical(d, rd_all, &differing);
				const double rl = compare(g, "drop", d, rd_all, n.residual_out, false);
				++drops_run;
				drops_detected += same ? 0 : 1;
				std::printf("PROBE graph %s drop k=%lld barriers=%lld detected=%d differing=%zu worst_rel_l2=%.3e [%s]\n",
						g, (long long)k, (long long)rk.barriers, same ? 0 : 1, differing, rl, rk.dropped.c_str());
			}
			set_env("GGML_RD_DROP_BARRIER", nullptr);
		} catch (const std::exception &e) {
			std::printf("PROBE graph %s: %s\n", g, e.what());
			ok = false;
		}
		ggml_backend_free(n.be);
		n.be = nullptr;
	}
	{
		Net n;
		n.be = rd_backend_or_null();
		n.rd = true;
		try {
			build(n, true);
			if (!allocate(n)) {
				throw std::runtime_error("allocation");
			}
			run(n, rd_f32, true);
			r_f32 = run(n, rd_f32, true);
		} catch (const std::exception &e) {
			std::printf("PROBE graph %s: %s\n", g, e.what());
			ok = false;
		}
		ggml_backend_free(n.be);
		n.be = nullptr;
	}

	// For the host oracle: both arms' outputs (and the block's input, for the branch norm).
	for (const auto &x : rd_nat) {
		g_dump.emplace_back("rd_native/" + x.first, x.second);
	}
	for (const auto &x : rd_f32) {
		if (x.first != "__residual_in") {
			g_dump.emplace_back("rd_f32/" + x.first, x.second);
		}
	}
	size_t dump_bytes = 0;
	for (const auto &x : g_dump) {
		dump_bytes += x.second.size() * 4;
	}

	// Info: the two RD arms against each other (the same weight values; the kernels differ).
	const double e_arms = compare(g, (std::string("rd_") + native + "_vs_rd_f32_info").c_str(), rd_nat, rd_f32, resid);
	size_t d_all = 0, d_again = 0;
	const bool same_all = identical(rd_nat, rd_all, &d_all);
	const bool same_again = identical(rd_nat, rd_again, &d_again);
	const bool fin = all_finite(rd_nat) && all_finite(rd_f32);
	const bool pass_ctl = drops_detected > 0;
	std::printf("PROBE graph %s SUMMARY weights=%s reference=host (dumped: %zu outputs, %.1f MiB) "
				"info rel_l2(rd_%s,rd_f32)=%.3e finite=%d elision_vs_barrier_all=%s (differing %zu) "
				"elision_repeat=%s (differing %zu) drop_control=%d/%d detected nodes=%d dispatches=%lld "
				"barriers=%lld/%lld status=%d/%d/%d/%d wall_s=%.1f%s\n",
			g, native.c_str(), g_dump.size(), dump_bytes / 1048576.0, native.c_str(), e_arms, fin ? 1 : 0,
			same_all ? "bit-identical" : "DIFFERENT", d_all, same_again ? "bit-identical" : "DIFFERENT", d_again,
			drops_detected, drops_run, nodes, (long long)r_el.dispatches, (long long)r_el.barriers,
			(long long)r_all.barriers, int(r_el.st), int(r_all.st), int(r_again.st), int(r_f32.st),
			(rdc::host_usec() - t_start) / 1e6,
			which == "sconv" ? (" L=" + std::to_string(shell_coords().size() / 3) + " neighbours=" +
									   std::to_string(g_sconv_neighbours) + "/" +
									   std::to_string(27 * shell_coords().size() / 3))
									  .c_str()
							 : "");
	return ok && fin && same_all && same_again && pass_ctl && r_el.st == GGML_STATUS_SUCCESS &&
			r_all.st == GGML_STATUS_SUCCESS && r_again.st == GGML_STATUS_SUCCESS && r_f32.st == GGML_STATUS_SUCCESS;
}

// --- G3.cost ---------------------------------------------------------------------------

struct CostRow {
	int64_t build_us = 0, alloc_us = 0, set_us = 0, compute_us = 0, read_us = 0, total_us = 0;
	int64_t frames = 0, nodes = 0, dispatches = 0, barriers = 0, gpu_ns = -1;
	ggml_rd_profile prof;
};

void print_profile_table(const char *what, const ggml_rd_profile &p, int64_t clock_us_x1000) {
	struct Agg {
		int64_t n = 0, pack = 0, record = 0, barriers = 0;
	};
	std::map<std::string, Agg> by;
	for (const auto &d : p.per_dispatch) {
		Agg &a = by[ggml_backend_rd_kernel_name(d.kernel)];
		++a.n;
		a.pack += d.pack_us;
		a.record += d.record_us;
		a.barriers += d.barrier ? 1 : 0;
	}
	std::vector<std::pair<std::string, Agg>> rows(by.begin(), by.end());
	std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second.pack + a.second.record > b.second.pack + b.second.record; });
	std::printf("PROBE cost %s profile2 dispatches=%lld total_us=%lld pack_us=%lld prepare_us=%lld upload_us=%lld "
				"record_us=%lld submit_us=%lld (per reading %.2f us; 3 readings per dispatch in this run)\n",
			what, (long long)p.dispatches, (long long)p.us_total, (long long)p.us_pack, (long long)p.us_prepare,
			(long long)p.us_upload, (long long)p.us_record, (long long)p.us_submit, clock_us_x1000 / 1000.0);
	for (const auto &r : rows) {
		std::printf("PROBE cost %s kernel=%-26s dispatches=%5lld barriers_before=%5lld pack_us/dispatch=%7.2f "
					"record_us/dispatch=%7.2f\n",
				what, r.first.c_str(), (long long)r.second.n, (long long)r.second.barriers,
				double(r.second.pack) / double(r.second.n), double(r.second.record) / double(r.second.n));
	}
}

// Microseconds per host clock reading, x1000 (1000 back-to-back readings).
int64_t clock_cost_x1000() {
	const int64_t t0 = rdc::host_usec();
	for (int i = 0; i < 1000; ++i) {
		(void)rdc::host_usec();
	}
	return rdc::host_usec() - t0;
}

// host_us: ggml-rd's graph_compute from entry to submit (GGML_RD_PROFILE=1);
// wait_us: the rest of ggml_backend_graph_compute, which is ggml's own
// synchronize: the WAIT_GPU yield, the next frame and the sync (the GPU time
// and a frame, not host work).
void print_cost_row(const char *what, int step, const CostRow &c) {
	std::printf("PROBE cost %s step=%d nodes=%lld dispatches=%lld barriers=%lld frames=%lld build_us=%lld "
				"alloc_us=%lld set_us=%lld host_us=%lld (pack %lld, prepare %lld, upload %lld, record %lld, "
				"submit %lld) host_us_per_node=%.2f host_us_per_dispatch=%.2f wait_us=%lld read_us=%lld "
				"step_total_us=%lld gpu_us=%.1f\n",
			what, step, (long long)c.nodes, (long long)c.dispatches, (long long)c.barriers, (long long)c.frames,
			(long long)c.build_us, (long long)c.alloc_us, (long long)c.set_us, (long long)c.prof.us_total,
			(long long)c.prof.us_pack, (long long)c.prof.us_prepare, (long long)c.prof.us_upload,
			(long long)c.prof.us_record, (long long)c.prof.us_submit,
			c.nodes ? double(c.prof.us_total) / double(c.nodes) : 0.0,
			c.dispatches ? double(c.prof.us_total) / double(c.dispatches) : 0.0,
			(long long)(c.compute_us - c.prof.us_total), (long long)c.read_us, (long long)c.total_us,
			c.gpu_ns >= 0 ? c.gpu_ns / 1000.0 : -1.0);
}

void print_cost_summary(const char *what, const std::vector<CostRow> &rows) {
	// Median over the timed steps (row 0 is the cold one).
	auto med = [&](auto f) {
		std::vector<double> v;
		for (size_t i = 1; i < rows.size(); ++i) {
			v.push_back(double(f(rows[i])));
		}
		std::sort(v.begin(), v.end());
		return v.empty() ? 0.0 : v[v.size() / 2];
	};
	const CostRow &c = rows.back();
	std::printf("PROBE cost %s SUMMARY steps=%zu nodes=%lld dispatches/graph=%lld barriers/graph=%lld "
				"frames/graph=%.0f (cold %lld) host_us=%.0f (cold %lld; pack %.0f prepare %.0f upload %.0f "
				"record %.0f submit %.0f) host_us/node=%.2f host_us/dispatch=%.2f build_us=%.0f alloc_us=%.0f "
				"wait_us=%.0f read_us=%.0f step_us=%.0f gpu_ms=%.2f\n",
			what, rows.size() - 1, (long long)c.nodes, (long long)c.dispatches, (long long)c.barriers,
			med([](const CostRow &r) { return r.frames; }), (long long)rows[0].frames,
			med([](const CostRow &r) { return r.prof.us_total; }), (long long)rows[0].prof.us_total,
			med([](const CostRow &r) { return r.prof.us_pack; }), med([](const CostRow &r) { return r.prof.us_prepare; }),
			med([](const CostRow &r) { return r.prof.us_upload; }), med([](const CostRow &r) { return r.prof.us_record; }),
			med([](const CostRow &r) { return r.prof.us_submit; }),
			med([](const CostRow &r) { return r.prof.us_total; }) / double(std::max<int64_t>(1, c.nodes)),
			med([](const CostRow &r) { return r.prof.us_total; }) / double(std::max<int64_t>(1, c.dispatches)),
			med([](const CostRow &r) { return r.build_us; }), med([](const CostRow &r) { return r.alloc_us; }),
			med([](const CostRow &r) { return r.compute_us - r.prof.us_total; }),
			med([](const CostRow &r) { return r.read_us; }), med([](const CostRow &r) { return r.total_us; }),
			med([](const CostRow &r) { return r.gpu_ns; }) / 1e6);
}

// Copy leaf `from` into leaf `to` on the GPU (buffer_copy).
void copy_leaf(Net &n, const std::string &from, const std::string &to) {
	ggml_backend_tensor_copy(n.named.at(from), n.named.at(to));
}

bool cost_decode(int steps) {
	using namespace app_graphs::qwen;
	ggml_backend_t be = rd_backend_or_null();
	if (be == nullptr) {
		return false;
	}
	Net w; // the model: weights and the KV caches, as the evaluator holds them
	w.be = be;
	w.rd = true;
	w.lctx = new_ctx(64 + 16 * layers);
	const int64_t cap = int64_t(kQwenPast + 2);
	leaf(w, "llm.tok.w", GGML_TYPE_F16, { hidden, vocab }, -0.05f, 0.05f);
	leaf(w, "llm.norm.w", GGML_TYPE_F32, { hidden }, 0.8f, 1.2f);
	matrix(w, "llm.out.w", GGML_TYPE_F16, false, { hidden, vocab }, hidden);
	std::vector<ggml_tensor *> keys, values;
	for (int l = 0; l < layers; ++l) {
		qwen_layer_weights(w, l, false);
		keys.push_back(leaf(w, "cache.k." + std::to_string(l), GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f));
		values.push_back(leaf(w, "cache.v." + std::to_string(l), GGML_TYPE_F32, { head_dim, kv_heads, cap, int64_t(kQwenBeams) }, -1.0f, 1.0f));
	}
	w.lbuf = ggml_backend_alloc_ctx_tensors(w.lctx, be);
	if (w.lbuf == nullptr) {
		std::printf("PROBE cost decode: weight allocation failed\n");
		return false;
	}
	// Random: the embedding, the head, layer 0 and its caches; the other layers are copies.
	std::vector<Leaf> first;
	for (const Leaf &l : w.leaves) {
		const bool per_layer = l.name.rfind("llm.l.", 0) == 0 || l.name.rfind("cache.", 0) == 0;
		const bool layer0 = l.name.rfind("llm.l.0.", 0) == 0 || l.name == "cache.k.0" || l.name == "cache.v.0";
		if (!per_layer || layer0) {
			first.push_back(l);
		}
	}
	fill_leaves(w, first);
	for (int l = 1; l < layers; ++l) {
		for (const char *s : { "an.w", "attn.q.w", "attn.k.w", "attn.v.w", "attn.q_norm.w", "attn.k_norm.w", "attn.o.w",
					 "fn.w", "mlp.g.w", "mlp.u.w", "mlp.d.w" }) {
			copy_leaf(w, std::string("llm.l.0.") + s, "llm.l." + std::to_string(l) + "." + s);
		}
		copy_leaf(w, "cache.k.0", "cache.k." + std::to_string(l));
		copy_leaf(w, "cache.v.0", "cache.v." + std::to_string(l));
	}
	ggml_backend_rd_ensure_idle();
	std::printf("PROBE cost decode model: %d layers, %.1f MiB of weights and caches on RD0\n", layers,
			ggml_backend_buffer_get_size(w.lbuf) / 1048576.0);

	const int64_t clock_x1000 = clock_cost_x1000();
	ggml_backend_rd_set_timestamps(true);
	std::vector<CostRow> rows;
	bool ok = true;
	const std::vector<size_t> slots = { 0, 1 };
	for (int step = 0; step <= steps + 1 && ok; ++step) {
		const bool prof2 = step == steps + 1; // the last: per-dispatch profile
		ggml_backend_rd_set_profile(prof2 ? 2 : 1);
		CostRow c;
		const int64_t f0 = frame_yields();
		const int64_t t0 = rdc::host_usec();
		// qwen_graph_evaluator::decode: a context, the graph, an allocator, every step.
		ggml_context *gctx = new_ctx(8192);
		ggml_tensor *ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, int64_t(kQwenBeams));
		ggml_tensor *position = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
		ggml_set_input(ids);
		ggml_set_input(position);
		ggml_tensor *logits = decode_step(gctx, ids, position, slots, kQwenPast, w.lookup(), keys, values, layers);
		ggml_cgraph *gf = ggml_new_graph_custom(gctx, 8192, false);
		ggml_build_forward_expand(gf, logits);
		const int64_t t1 = rdc::host_usec();
		ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
		if (!ggml_gallocr_reserve(galloc, gf) || !ggml_gallocr_alloc_graph(galloc, gf)) {
			std::printf("PROBE cost decode: allocation failed\n");
			ok = false;
		}
		const int64_t t2 = rdc::host_usec();
		const int32_t id_v[2] = { 1000 + step, 2000 + step }, pos_v = int32_t(kQwenPast);
		ggml_backend_tensor_set(ids, id_v, 0, sizeof id_v);
		ggml_backend_tensor_set(position, &pos_v, 0, sizeof pos_v);
		const int64_t t3 = rdc::host_usec();
		const ggml_status st = ok ? ggml_backend_graph_compute(be, gf) : GGML_STATUS_FAILED;
		const int64_t t4 = rdc::host_usec();
		c.prof = ggml_backend_rd_last_profile();
		ggml_backend_rd_last_graph(&c.dispatches, &c.barriers);
		std::vector<float> out(size_t(vocab) * kQwenBeams);
		ggml_backend_tensor_get(logits, out.data(), 0, out.size() * 4); // WAIT_GPU, then the sync
		const int64_t t5 = rdc::host_usec();
		c.gpu_ns = ggml_backend_rd_last_gpu_ns();
		c.frames = frame_yields() - f0;
		c.build_us = t1 - t0;
		c.alloc_us = t2 - t1;
		c.set_us = t3 - t2;
		c.compute_us = t4 - t3;
		c.read_us = t5 - t4;
		c.total_us = t5 - t0;
		c.nodes = ggml_graph_n_nodes(gf);
		ok = ok && st == GGML_STATUS_SUCCESS && finite(out);
		if (!finite(out)) {
			std::printf("PROBE cost decode: non-finite logits\n");
		}
		ggml_gallocr_free(galloc);
		ggml_free(gctx);
		if (prof2) {
			print_cost_row("decode(profile2)", step, c);
			print_profile_table("decode", c.prof, clock_x1000);
		} else {
			print_cost_row("decode", step, c);
			rows.push_back(c);
		}
	}
	ggml_backend_rd_set_profile(0);
	ggml_backend_rd_set_timestamps(false);
	if (!rows.empty()) {
		print_cost_summary("decode", rows);
	}
	std::printf("PROBE cost decode host_clock_us_per_reading=%.2f\n", clock_x1000 / 1000.0);
	ggml_backend_buffer_free(w.lbuf);
	w.lbuf = nullptr;
	ggml_backend_free(be);
	return ok;
}

bool cost_dit(int steps) {
	using namespace app_graphs::dit;
	ggml_backend_t be = rd_backend_or_null();
	if (be == nullptr) {
		return false;
	}
	const Hparams hp;
	Net w;
	w.be = be;
	w.rd = true;
	w.lctx = new_ctx(64 + 40 * hp.num_blocks);
	const int64_t C = hp.model_channels, N = hp.tokens();
	auto lin = [&](const std::string &name, ggml_type t, int64_t in, int64_t out) {
		matrix(w, name + ".weight", t, false, { in, out }, in);
		const float bb = 1.0f / std::sqrt(float(in));
		leaf(w, name + ".bias", GGML_TYPE_F32, { out }, -bb, bb);
	};
	lin("input_layer", GGML_TYPE_F16, hp.in_channels, C);
	lin("t_embedder.mlp.0", GGML_TYPE_F16, 256, C);
	lin("t_embedder.mlp.2", GGML_TYPE_F16, C, C);
	lin("adaLN_modulation.1", GGML_TYPE_F32, C, 6 * C);
	lin("out_layer", GGML_TYPE_F16, C, hp.out_channels);
	for (int b = 0; b < hp.num_blocks; ++b) {
		dit_block_weights(w, b, hp, false);
	}
	const Leaves in = dit_leaves(w, hp);
	ggml_tensor *x_t = leaf(w, "x", GGML_TYPE_F32, { N, hp.in_channels }, -1.0f, 1.0f);
	const std::vector<float> te = timestep_embedding(0.75f * 1000.0f, 256);
	ggml_tensor *temb = custom_leaf(w, "temb", GGML_TYPE_F32, { 256 },
			[te](std::vector<uint8_t> &b) { std::memcpy(b.data(), te.data(), b.size()); });
	w.lbuf = ggml_backend_alloc_ctx_tensors(w.lctx, be);
	if (w.lbuf == nullptr) {
		std::printf("PROBE cost dit: weight allocation failed\n");
		return false;
	}
	std::vector<Leaf> first;
	for (const Leaf &l : w.leaves) {
		if (l.name.rfind("blocks.", 0) != 0 || l.name.rfind("blocks.0.", 0) == 0) {
			first.push_back(l);
		}
	}
	fill_leaves(w, first);
	for (int b = 1; b < hp.num_blocks; ++b) {
		for (const Leaf &l : w.leaves) {
			if (l.name.rfind("blocks.0.", 0) == 0) {
				copy_leaf(w, l.name, "blocks." + std::to_string(b) + "." + l.name.substr(9));
			}
		}
	}
	ggml_backend_rd_ensure_idle();
	std::printf("PROBE cost dit model: %d blocks, %lld tokens, %.1f MiB of weights on RD0 (bf16 matrices)\n",
			hp.num_blocks, (long long)N, ggml_backend_buffer_get_size(w.lbuf) / 1048576.0);

	const int64_t clock_x1000 = clock_cost_x1000();
	ggml_backend_rd_set_timestamps(true);
	std::vector<CostRow> rows;
	bool ok = true;
	for (int step = 0; step <= steps + 1 && ok; ++step) {
		const bool prof2 = step == steps + 1;
		ggml_backend_rd_set_profile(prof2 ? 2 : 1);
		CostRow c;
		const int64_t f0 = frame_yields();
		const int64_t t0 = rdc::host_usec();
		// trellis2_ss_flow_forward: a context, the graph, an allocator, every call.
		ggml_context *gctx = new_ctx(32768);
		ggml_cgraph *gf = ggml_new_graph_custom(gctx, 32768, false);
		ggml_tensor *y = nullptr;
		try {
			y = forward(gctx, x_t, temb, hp, w.lookup(), in);
		} catch (const std::exception &e) {
			std::printf("PROBE cost dit: %s\n", e.what());
			ggml_free(gctx);
			ok = false;
			break;
		}
		ggml_set_output(y);
		ggml_build_forward_expand(gf, y);
		const int64_t t1 = rdc::host_usec();
		ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
		if (!ggml_gallocr_alloc_graph(galloc, gf)) {
			std::printf("PROBE cost dit: allocation failed\n");
			ok = false;
		}
		const int64_t t2 = rdc::host_usec();
		const int64_t t3 = t2; // the inputs are leaves here, set once
		const ggml_status st = ok ? ggml_backend_graph_compute(be, gf) : GGML_STATUS_FAILED;
		const int64_t t4 = rdc::host_usec();
		c.prof = ggml_backend_rd_last_profile();
		ggml_backend_rd_last_graph(&c.dispatches, &c.barriers);
		std::vector<float> out(size_t(ggml_nelements(y)));
		ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
		const int64_t t5 = rdc::host_usec();
		c.gpu_ns = ggml_backend_rd_last_gpu_ns();
		c.frames = frame_yields() - f0;
		c.build_us = t1 - t0;
		c.alloc_us = t2 - t1;
		c.set_us = t3 - t2;
		c.compute_us = t4 - t3;
		c.read_us = t5 - t4;
		c.total_us = t5 - t0;
		c.nodes = ggml_graph_n_nodes(gf);
		ok = ok && st == GGML_STATUS_SUCCESS && finite(out);
		if (!finite(out)) {
			std::printf("PROBE cost dit: non-finite output\n");
		}
		ggml_gallocr_free(galloc);
		ggml_free(gctx);
		if (prof2) {
			print_cost_row("dit(profile2)", step, c);
			print_profile_table("dit", c.prof, clock_x1000);
		} else {
			print_cost_row("dit", step, c);
			rows.push_back(c);
		}
	}
	ggml_backend_rd_set_profile(0);
	ggml_backend_rd_set_timestamps(false);
	if (!rows.empty()) {
		print_cost_summary("dit", rows);
	}
	std::printf("PROBE cost dit host_clock_us_per_reading=%.2f\n", clock_x1000 / 1000.0);
	ggml_backend_buffer_free(w.lbuf);
	w.lbuf = nullptr;
	ggml_backend_free(be);
	return ok;
}

} // namespace

// arg: "qwen", "dit", "dit:8" (8^3 tokens), "sconv".
bool graph_probe(const std::string &arg) {
	const size_t colon = arg.find(':');
	const std::string which = arg.substr(0, colon);
	const int res = colon == std::string::npos ? 0 : std::atoi(arg.c_str() + colon + 1);
	return graph_one(which, res);
}

// arg: "decode[:steps]" or "dit[:steps]".
bool cost_probe(const std::string &arg) {
	const size_t colon = arg.find(':');
	const std::string which = arg.substr(0, colon);
	const int steps = colon == std::string::npos ? 5 : std::max(1, std::atoi(arg.c_str() + colon + 1));
	for (const char *k : { "GGML_RD_BARRIER_ALL", "GGML_RD_DROP_BARRIER", "GGML_RD_FAULT", "GGML_RD_PROFILE" }) {
		unsetenv(k);
	}
	try {
		if (which == "decode") {
			return cost_decode(steps);
		}
		if (which == "dit") {
			return cost_dit(steps);
		}
	} catch (const std::exception &e) {
		std::printf("PROBE cost %s: %s\n", which.c_str(), e.what());
		return false;
	}
	std::printf("PROBE cost: unknown '%s' (decode, dit)\n", which.c_str());
	return false;
}

// The last G3.graph run's outputs, for the host oracle: "name bytes" per line.
std::string graph_dump_list() {
	std::string r;
	for (const auto &x : g_dump) {
		r += x.first + " " + std::to_string(x.second.size() * 4) + "\n";
	}
	return r;
}

// Bytes [offset, offset + bytes) of dump entry `index` (clamped to the entry);
// nullptr past the end.
const uint8_t *graph_dump_chunk(size_t index, size_t offset, size_t &bytes) {
	if (index >= g_dump.size()) {
		bytes = 0;
		return nullptr;
	}
	const size_t size = g_dump[index].second.size() * 4;
	if (offset >= size) {
		bytes = 0;
		return nullptr;
	}
	bytes = std::min(bytes, size - offset);
	return reinterpret_cast<const uint8_t *>(g_dump[index].second.data()) + offset;
}

} // namespace probes
