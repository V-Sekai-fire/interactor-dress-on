// fit.elf -- the fit stage: cloth-fit's garment retargeting (PolyFEM,
// polysolve, ipc-toolkit, libigl; the SDF spline sampler a Lean kernel) in
// the guest, one phase per vmcall.
//
// One ELF per stage (AGENTS.md rule 6); the only ELF with Eigen in it (rule 3).
// Inputs arrive as packed arrays and a JSON string (the guest has no files;
// every open is counted and refused, fit_tools.cpp). The host sets the body,
// skeletons, optional skin weights, garment and config, then calls fit_begin
// and fit_step once per phase (2 * incremental_steps phases). Long phases are
// meant for a GDScript worker Thread (Gate 0F probe 7) with execution_timeout
// raised; nothing here waits on anything.
//
// Frames: "body space" is the target avatar's frame as the host gave it;
// "normalised" is the solve frame (upstream's step_garment_*.obj frame),
// x_solve = target_scale * (M x_body) + center (fit::Normalisation).
//
// Every entry point catches and answers "FAIL: ..." instead of throwing into
// the host. Every one has a no-argument wrapper in project/main.gd (rule 8).

#include <api.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fit_broad_phase.h"
#include "fit_driver.h"
#include "fit_gpu.h"
#include "fit_probes.h"
#include "fit_sim_hessian.h"
#include "fit_tools.h"

namespace {

std::unique_ptr<fit::FitDriver> g_drv;
fit::FitInput g_in;
bool g_have_body = false, g_have_skel = false, g_have_garment = false, g_have_config = false;
int g_io_at_begin = -1; // io_attempts when fit_begin started; -1 before

// Cut 6g-C: where SimilarityForm's Hessian is assembled. 0 is the CPU path
// (bitwise what fit.elf did before this cut: the flat control), 1 the Lean
// kernels on the worker's RenderingDevice, 2 their cpp twin on the guest
// CPU (the gate's second target). Default on, gated (gates/6g-polyfem-gpu/c1).
int g_gpu_mode = 1;
float g_gpu_sign = 1.0f; // the gate's flipped-sign control scales every block
std::unique_ptr<fit_gpu::SimProblem> g_sim;
fit_gpu::GpuHessian g_gpu; // its device outlives sessions; freed by fit_gpu_close
int64_t g_twin_us = 0, g_twin_calls = 0;

// Cut 6g-C, stage 2: the CCD broad phase. 0 the CPU path (ipc-toolkit's
// BVH), 1 the Lean kernels on the device, 2 the same audited by the CPU
// build after every line search (the C2 gate's arm).
int g_bp_mode = 1;
fit_gpu::GpuBroadPhase g_bp;
std::string g_bp_open_error;

bool bp_hook(const ipc::CollisionMesh &mesh, const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, double inflation, ipc::Candidates &out) {
	if (g_bp_mode == 0 || !g_drv)
		return false;
	if (!g_bp.is_open()) {
		std::string err;
		if (!g_gpu.open_device(&err) || !g_bp.open(g_gpu.device(), mesh, g_drv->avatar_vertex_count(), g_drv->self_collision(), &err)) {
			g_bp_open_error = err;
			return false;
		}
		auto cf = g_drv->contact_form();
		g_bp.set_audit(g_bp_mode == 2, cf ? cf->broad_phase_method() : ipc::BroadPhaseMethod::BVH);
	}
	return g_bp.build(V0, V1, inflation, out);
}

std::string bp_text() {
	const fit_gpu::BpStats &s = g_bp.stats();
	char b[640];
	std::snprintf(b, sizeof b, "broad mode %d builds %lld fallbacks %lld mean_us %.0f (gpu %.0f convert %.0f) pairs_mean %.0f last_pairs %lld queries %u grown %lld",
			g_bp_mode, (long long)s.builds, (long long)s.fallbacks, s.builds ? double(s.total_us) / double(s.builds) : 0.0,
			s.builds ? double(s.gpu_us) / double(s.builds) : 0.0, s.builds ? double(s.convert_us) / double(s.builds) : 0.0,
			s.builds ? double(s.pairs) / double(s.builds) : 0.0, (long long)s.last_pairs, g_bp.queries(), (long long)s.grown);
	std::string t = b;
	if (s.audit_builds) {
		std::snprintf(b, sizeof b, " | audit builds %lld cpu_mean_us %.0f cpu_pairs_mean %.0f MISSED %lld extra %lld (%.2f%%)",
				(long long)s.audit_builds, double(s.audit_cpu_us) / double(s.audit_builds), double(s.audit_cpu_pairs) / double(s.audit_builds),
				(long long)s.audit_missed, (long long)s.audit_extra, s.audit_cpu_pairs ? 100.0 * double(s.audit_extra) / double(s.audit_cpu_pairs) : 0.0);
		t += b;
	}
	if (!s.last_error.empty())
		t += " last_error: " + s.last_error;
	if (!g_bp_open_error.empty())
		t += " open_error: " + g_bp_open_error;
	return t;
}
int64_t g_gpu_fallbacks_at_step = 0;
std::string g_gpu_open_error;

bool gpu_hook(const Eigen::VectorXd &x, bool psd, polyfem::StiffnessMatrix &out) {
	if (!g_sim)
		return false;
	if (g_gpu_mode == 2) {
		const int64_t t0 = rdc::host_usec();
		const bool ok = fit_gpu::SimTwin::assemble(*g_sim, x, psd, g_gpu_sign, out);
		g_twin_us += rdc::host_usec() - t0;
		g_twin_calls++;
		return ok;
	}
	if (g_gpu_mode == 1) {
		if (!g_gpu.is_open()) {
			std::string err;
			if (!g_gpu.open(*g_sim, &err)) {
				g_gpu_open_error = err;
				return false;
			}
		}
		return g_gpu.assemble(*g_sim, x, psd, g_gpu_sign, out);
	}
	return false;
}

// After begin(): the session data for the kernels and the hook on the form.
void attach_gpu(fit::FitDriver &d) {
	g_sim.reset();
	auto form = d.similarity_form();
	if (!form)
		return;
	if (g_gpu_mode == 0) {
		form->set_hessian_hook(nullptr);
		return;
	}
	g_sim = std::make_unique<fit_gpu::SimProblem>();
	g_sim->build(*form);
	form->set_hessian_hook(&gpu_hook);
}

void attach_bp(fit::FitDriver &d) {
	auto cf = d.contact_form();
	if (!cf)
		return;
	if (g_bp_mode == 0) {
		cf->set_broad_phase_hook(nullptr);
		return;
	}
	cf->set_broad_phase_hook(&bp_hook);
	if (g_bp.is_open())
		g_bp.set_audit(g_bp_mode == 2, cf->broad_phase_method());
}

std::string gpu_text() {
	const fit_gpu::GpuStats &s = g_gpu.stats();
	char b[640];
	std::snprintf(b, sizeof b, "gpu mode %d assemblies %lld fallbacks %lld mean_us %.0f (upload %.0f gpu %.0f readback %.0f) gpu_ms_total %.2f worker_syncs %lld same_frame_syncs %lld twin_calls %lld twin_mean_us %.0f%s%s",
			g_gpu_mode, (long long)s.assemblies, (long long)s.fallbacks,
			s.assemblies ? double(s.total_us) / double(s.assemblies) : 0.0,
			s.assemblies ? double(s.upload_us) / double(s.assemblies) : 0.0,
			s.assemblies ? double(s.gpu_us) / double(s.assemblies) : 0.0,
			s.assemblies ? double(s.readback_us) / double(s.assemblies) : 0.0,
			double(s.gpu_ns) / 1e6, (long long)g_gpu.device().worker_syncs(), (long long)g_gpu.device().same_frame_syncs(),
			(long long)g_twin_calls, g_twin_calls ? double(g_twin_us) / double(g_twin_calls) : 0.0,
			s.last_error.empty() ? "" : " last_error: ", s.last_error.c_str());
	std::string t = b;
	if (!g_gpu_open_error.empty())
		t += " open_error: " + g_gpu_open_error;
	return t;
}
int g_newton_total = 0;
int g_preview_every = 1;
fit::PhaseStats g_last;
uint64_t g_last_instructions = 0; // instructions retired by the last fit_step / fit_run_all

// libriscv's instruction counter through the instret CSR (Gate 6.P: the
// execution_timeout a phase needs). It counts from the start of the vmcall,
// so the difference across a call is that call's instructions.
uint64_t instret() {
	uint64_t v;
	asm volatile("rdinstret %0" : "=r"(v));
	return v;
}

Variant text(const std::string &s) {
	return Variant(String(s));
}

Variant fail(const std::string &what) {
	return text("FAIL: " + what);
}

template <typename F>
Variant guarded(const char *name, F &&f) {
	try {
		return f();
	} catch (const std::exception &e) {
		return fail(std::string(name) + ": " + e.what());
	} catch (...) {
		return fail(std::string(name) + ": unknown exception");
	}
}

template <typename T>
std::vector<double> to_f64(const std::vector<T> &v) {
	return std::vector<double>(v.begin(), v.end());
}

std::vector<int> to_int(const std::vector<int32_t> &v) {
	return std::vector<int>(v.begin(), v.end());
}

void check_triples(const char *what, size_t n) {
	if (n % 3)
		throw std::runtime_error(std::string(what) + " must be triples (size " + std::to_string(n) + ")");
}

void check_indices(const char *what, const std::vector<int> &idx, size_t n_vertices) {
	for (int i : idx)
		if (i < 0 || size_t(i) >= n_vertices)
			throw std::runtime_error(std::string(what) + ": index " + std::to_string(i) + " out of range (" +
					std::to_string(n_vertices) + " vertices)");
}

fit::FitDriver &driver() {
	if (!g_drv)
		throw std::runtime_error("fit_begin has not run");
	return *g_drv;
}

// Body space (host frame) -> solve frame: x_solve = scale * (M x) + center.
std::vector<double> body_to_solve(const std::vector<float> &v, const fit::Normalisation &nm) {
	std::vector<double> out(v.size());
	const double *M = nm.to_canonical;
	for (size_t i = 0; i + 2 < v.size(); i += 3) {
		const double x = v[i], y = v[i + 1], z = v[i + 2];
		for (int r = 0; r < 3; r++)
			out[i + r] = nm.target_scale * (M[r * 3 + 0] * x + M[r * 3 + 1] * y + M[r * 3 + 2] * z) + nm.center[r];
	}
	return out;
}

// Solve frame -> body space (fit_driver's garment_avatar_frame, for any array).
std::vector<float> solve_to_body_f32(const std::vector<double> &v, const fit::Normalisation &nm) {
	std::vector<float> out(v.size());
	const double *M = nm.to_canonical;
	for (size_t i = 0; i + 2 < v.size(); i += 3) {
		double c[3];
		for (int j = 0; j < 3; j++)
			c[j] = (v[i + j] - nm.center[j]) / nm.target_scale;
		for (int j = 0; j < 3; j++)
			out[i + j] = float(M[0 * 3 + j] * c[0] + M[1 * 3 + j] * c[1] + M[2 * 3 + j] * c[2]);
	}
	return out;
}

std::string heap_text() {
	const fit::HeapInfo h = fit::heap_info();
	if (!h.ok)
		return "heap n/a";
	char b[128];
	std::snprintf(b, sizeof b, "heap_used %.1f MiB heap_free %.1f MiB chunks %llu", h.bytes_used / 1048576.0,
			h.bytes_free / 1048576.0, (unsigned long long)h.chunks_used);
	return b;
}

std::string phase_text(const fit::PhaseStats &st) {
	char b[512];
	std::snprintf(b, sizeof b, "phase %d (substep %d, %s): newton %d minimize %d post_steps %d energy %.17g |grad| %.6g status %s",
			st.phase, st.substep, st.kind == 0 ? "AL" : "reduced", st.newton_iterations, st.minimize_calls, st.post_steps,
			st.energy, st.grad_norm, st.status.c_str());
	return b;
}

int io_in_window() {
	return g_io_at_begin < 0 ? 0 : fit::io_attempts() - g_io_at_begin;
}

// --- inputs ------------------------------------------------------------------------

Variant fit_reset() {
	return guarded("fit_reset", [] {
		g_drv.reset();
		g_sim.reset();
		g_in = fit::FitInput();
		g_have_body = g_have_skel = g_have_garment = g_have_config = false;
		g_io_at_begin = -1;
		g_newton_total = 0;
		g_last = fit::PhaseStats();
		return text("OK reset");
	});
}

Variant fit_set_body(PackedFloat32Array v, PackedInt32Array f) {
	return guarded("fit_set_body", [&] {
		std::vector<float> vv = v.fetch();
		std::vector<int> ff = to_int(f.fetch());
		check_triples("body vertices", vv.size());
		check_triples("body faces", ff.size());
		check_indices("body faces", ff, vv.size() / 3);
		g_in.avatar_v = to_f64(vv);
		g_in.avatar_f = std::move(ff);
		g_have_body = true;
		return text("OK body " + std::to_string(vv.size() / 3) + " v / " + std::to_string(g_in.avatar_f.size() / 3) + " f");
	});
}

Variant fit_set_skeletons(PackedFloat32Array src, PackedFloat32Array tgt, PackedInt32Array bones) {
	return guarded("fit_set_skeletons", [&] {
		std::vector<float> s = src.fetch(), t = tgt.fetch();
		std::vector<int> b = to_int(bones.fetch());
		check_triples("source skeleton", s.size());
		check_triples("target skeleton", t.size());
		if (s.size() != t.size())
			throw std::runtime_error("source and target skeletons differ in joint count");
		if (b.size() % 2)
			throw std::runtime_error("bones must be index pairs");
		check_indices("bones", b, s.size() / 3);
		g_in.skeleton_v = to_f64(s);
		g_in.target_skeleton_v = to_f64(t);
		g_in.skeleton_b = b;
		g_in.target_skeleton_b = b;
		g_have_skel = true;
		return text("OK skeletons " + std::to_string(s.size() / 3) + " joints / " + std::to_string(b.size() / 2) + " bones");
	});
}

Variant fit_set_skin_weights(PackedFloat32Array w) {
	return guarded("fit_set_skin_weights", [&] {
		g_in.target_avatar_skinning_weights = to_f64(w.fetch());
		return text("OK skin weights " + std::to_string(g_in.target_avatar_skinning_weights.size()) +
				" (n_bones x n_body_v, column-major; empty = distance projection)");
	});
}

Variant fit_set_garment(PackedFloat32Array v, PackedInt32Array f, PackedInt32Array nofit) {
	return guarded("fit_set_garment", [&] {
		std::vector<float> vv = v.fetch();
		std::vector<int> ff = to_int(f.fetch());
		std::vector<int> nf = to_int(nofit.fetch());
		check_triples("garment vertices", vv.size());
		check_triples("garment faces", ff.size());
		check_indices("garment faces", ff, vv.size() / 3);
		check_indices("no-fit vertices", nf, vv.size() / 3);
		g_in.garment_v = to_f64(vv);
		g_in.garment_f = std::move(ff);
		g_in.no_fit_vertices = std::move(nf);
		g_have_garment = true;
		return text("OK garment " + std::to_string(vv.size() / 3) + " v / " + std::to_string(g_in.garment_f.size() / 3) +
				" f, no-fit " + std::to_string(g_in.no_fit_vertices.size()));
	});
}

Variant fit_set_config(String json) {
	return guarded("fit_set_config", [&] {
		g_in.setup_json = fit::strip_path_keys(json.utf8());
		g_have_config = true;
		return text("OK config " + std::to_string(g_in.setup_json.size()) + " bytes");
	});
}

// --- the solve ---------------------------------------------------------------------

Variant fit_begin() {
	return guarded("fit_begin", [] {
		std::string missing;
		if (!g_have_body) missing += " body";
		if (!g_have_skel) missing += " skeletons";
		if (!g_have_garment) missing += " garment";
		if (!g_have_config) missing += " config";
		if (!missing.empty())
			return fail("fit_begin: not set:" + missing);
		g_drv = std::make_unique<fit::FitDriver>();
		g_drv->set_preview_every(g_preview_every);
		g_newton_total = 0;
		g_last = fit::PhaseStats();
		g_io_at_begin = fit::io_attempts();
		std::string err;
		if (!g_drv->begin(g_in, &err)) {
			const auto &ids = g_drv->start_intersections().ids;
			std::string s = "fit_begin: " + err;
			if (ids[0] >= 0) {
				char b[128];
				std::snprintf(b, sizeof b, " (start intersects: edge (%d,%d) face (%d,%d,%d))", ids[0], ids[1], ids[2], ids[3], ids[4]);
				s += b;
			}
			g_drv.reset();
			return fail(s);
		}
		attach_gpu(*g_drv);
		attach_bp(*g_drv);
		const fit::Normalisation &nm = g_drv->normalisation();
		char b[480];
		std::snprintf(b, sizeof b, "OK begin: phases %d, target_scale %.17g center (%.17g, %.17g, %.17g), source_scale %.17g, io_attempts %d, gpu mode %d%s",
				g_drv->phase_count(), nm.target_scale, nm.center[0], nm.center[1], nm.center[2], nm.source_scale, io_in_window(),
				g_gpu_mode, g_sim ? (" (hinges " + std::to_string(g_sim->n_hinges) + " slots " + std::to_string(g_sim->n_slots) +
											" nnz " + std::to_string(g_sim->nnz()) + " session_data " + std::to_string(g_sim->bytes() / 1024) + " KiB)").c_str()
									: "");
		return text(b);
	});
}

Variant fit_step() {
	return guarded("fit_step", [] {
		fit::FitDriver &d = driver();
		if (d.failed())
			return fail("fit_step: the driver failed earlier: " + g_last.error);
		if (d.done())
			return text("DONE phases " + std::to_string(d.phase_count()) + " newton " + std::to_string(g_newton_total));
		fit::PhaseStats st;
		const uint64_t i0 = instret();
		const int64_t fb0 = g_gpu.stats().fallbacks;
		const int64_t as0 = g_gpu.stats().assemblies, us0 = g_gpu.stats().total_us;
		const int64_t bb0 = g_bp.stats().builds, bu0 = g_bp.stats().total_us, bf0 = g_bp.stats().fallbacks;
		const int64_t bc0 = g_bp.stats().audit_cpu_us, bm0 = g_bp.stats().audit_missed, bx0 = g_bp.stats().audit_extra;
		const bool ok = d.step(&st);
		g_last_instructions = instret() - i0;
		g_last = st;
		g_newton_total += st.newton_iterations;
		if (!ok)
			return fail("fit_step: " + phase_text(st) + ": " + st.error);
		const int64_t fb = g_gpu.stats().fallbacks - fb0;
		char gb[320];
		std::snprintf(gb, sizeof gb, " gpu_assemblies %lld gpu_us %lld gpu_fallbacks %lld bp_builds %lld bp_us %lld bp_cpu_us %lld bp_missed %lld bp_extra %lld bp_fallbacks %lld",
				(long long)(g_gpu.stats().assemblies - as0), (long long)(g_gpu.stats().total_us - us0), (long long)fb,
				(long long)(g_bp.stats().builds - bb0), (long long)(g_bp.stats().total_us - bu0), (long long)(g_bp.stats().audit_cpu_us - bc0),
				(long long)(g_bp.stats().audit_missed - bm0), (long long)(g_bp.stats().audit_extra - bx0), (long long)(g_bp.stats().fallbacks - bf0));
		// The GPU path was asked for and fell back to the CPU inside the phase:
		// loud, not silent (the phase's result stands; fit_gpu_stats has the reason).
		if (g_gpu_mode == 1 && (fb > 0 || !g_gpu_open_error.empty()))
			return fail("fit_step: the GPU Hessian fell back to the CPU path: " + gpu_text() + " | " + phase_text(st));
		if (g_bp_mode != 0 && (g_bp.stats().fallbacks - bf0 > 0 || !g_bp_open_error.empty()))
			return fail("fit_step: the GPU broad phase fell back to the CPU path: " + bp_text() + " | " + phase_text(st));
		return text("OK " + phase_text(st) + " io_attempts " + std::to_string(io_in_window()) + " instructions " +
				std::to_string(g_last_instructions) + " " + heap_text() + gb);
	});
}

Variant fit_run_all() {
	return guarded("fit_run_all", [] {
		fit::FitDriver &d = driver();
		std::string log;
		const uint64_t i0 = instret();
		while (!d.done() && !d.failed()) {
			fit::PhaseStats st;
			const bool ok = d.step(&st);
			g_last = st;
			g_newton_total += st.newton_iterations;
			log += (ok ? "OK " : "FAIL ") + phase_text(st) + (ok ? "" : ": " + st.error) + "\n";
			if (!ok)
				return fail("fit_run_all:\n" + log);
		}
		g_last_instructions = instret() - i0;
		return text(log + "DONE phases " + std::to_string(d.phase_count()) + " newton " + std::to_string(g_newton_total) +
				" io_attempts " + std::to_string(io_in_window()) + " instructions " + std::to_string(g_last_instructions));
	});
}

Variant fit_status() {
	return guarded("fit_status", [] {
		char b[512];
		if (!g_drv) {
			std::snprintf(b, sizeof b, "phase -/- newton 0 energy - io_attempts %d io_total %d %s inputs body=%d skel=%d garment=%d config=%d",
					io_in_window(), fit::io_attempts(), heap_text().c_str(), g_have_body, g_have_skel, g_have_garment, g_have_config);
			return text(b);
		}
		const fit::FitDriver &d = *g_drv;
		std::snprintf(b, sizeof b, "phase %d/%d%s newton %d energy %.17g |grad| %.6g status %s io_attempts %d io_total %d last_instructions %llu %s",
				d.next_phase(), d.phase_count(), d.failed() ? " FAILED" : d.done() ? " done" : "", g_newton_total, g_last.energy,
				g_last.grad_norm, g_last.status.empty() ? "-" : g_last.status.c_str(), io_in_window(), fit::io_attempts(),
				(unsigned long long)g_last_instructions, heap_text().c_str());
		return text(b);
	});
}

// --- results -----------------------------------------------------------------------

Variant fit_result_vertices() {
	return guarded("fit_result_vertices", [] {
		std::vector<double> v;
		driver().garment_avatar_frame(&v);
		std::vector<float> f(v.begin(), v.end());
		return Variant(PackedFloat32Array(f));
	});
}

Variant fit_result_vertices_f64() {
	return guarded("fit_result_vertices_f64", [] {
		std::vector<double> v;
		driver().garment_solve_frame(&v);
		return Variant(PackedFloat64Array(v));
	});
}

// k > 0 sets the preview interval (every k-th post_step, i.e. Newton
// iteration, writes the ring; it applies from the next fit_begin), k <= 0
// leaves it. Returns the newest snapshot in body space (empty if none).
Variant fit_preview(int k) {
	return guarded("fit_preview", [&] {
		if (k > 0) {
			g_preview_every = k;
			if (g_drv)
				g_drv->set_preview_every(k);
		}
		if (!g_drv)
			return Variant(PackedFloat32Array(std::vector<float>()));
		std::vector<double> v;
		if (g_drv->latest_preview(&v) < 0)
			return Variant(PackedFloat32Array(std::vector<float>()));
		return Variant(PackedFloat32Array(solve_to_body_f32(v, g_drv->normalisation())));
	});
}

// ipc::my_has_intersections on the avatar at the current phase's alpha and the
// current garment, or, if override is non-empty, that garment instead (body
// space, same vertex order).
Variant fit_check_intersections(PackedFloat32Array override_v) {
	return guarded("fit_check_intersections", [&] {
		fit::FitDriver &d = driver();
		std::vector<float> ov = override_v.fetch();
		fit::IntersectionReport r;
		if (ov.empty()) {
			r = d.check_intersections();
		} else {
			if (ov.size() != size_t(d.garment_vertex_count()) * 3)
				throw std::runtime_error("override has " + std::to_string(ov.size() / 3) + " vertices, the garment " +
						std::to_string(d.garment_vertex_count()));
			r = d.check_intersections_with(body_to_solve(ov, d.normalisation()));
		}
		if (!r.intersects())
			return text("OK none");
		char b[160];
		std::snprintf(b, sizeof b, "OK INTERSECTS edge (%d,%d) face (%d,%d,%d)", r.ids[0], r.ids[1], r.ids[2], r.ids[3], r.ids[4]);
		return text(b);
	});
}

// The intersection check's positive control: the current garment (body space)
// with one vertex moved `dist` solve units (the solve frame is the garment's
// metric frame, voxel 0.01 = 1 cm) into the avatar along -grad SDF. The vertex
// is the one with the smallest SDF value (the closest to the body, through the
// Lean sampler). dist = 0 gives the unpushed garment through the same f32
// round trip, the flat control. Hand the result to fit_check_intersections.
Variant fit_push_vertex(double dist) {
	return guarded("fit_push_vertex", [&] {
		fit::FitDriver &d = driver();
		std::vector<double> g;
		d.garment_solve_frame(&g);
		double h = 0;
		const std::vector<double> s = fit::sdf_sample(g, &h);
		const size_t n = g.size() / 3;
		size_t best = 0;
		for (size_t i = 1; i < n; i++)
			if (s[10 * i] < s[10 * best])
				best = i;
		double gr[3] = {s[10 * best + 1], s[10 * best + 2], s[10 * best + 3]};
		const double len = std::sqrt(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
		if (!(len > 0))
			throw std::runtime_error("zero SDF gradient at the chosen vertex");
		for (int j = 0; j < 3; j++)
			g[3 * best + j] -= dist * gr[j] / len;
		return Variant(PackedFloat32Array(solve_to_body_f32(g, d.normalisation())));
	});
}

// The SDF grid sampled at points (solve frame, xyz): 10 doubles per point,
// (x, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz). x is a distance in solve
// units (the grid stores distances, clamped to [-h, 150h]); g and h are
// per index step, so g / h and h / h^2 are the solve-frame derivatives, as
// FitForm scales them.
Variant fit_sdf_dump(PackedFloat64Array pts) {
	return guarded("fit_sdf_dump", [&] {
		double h = 0;
		std::vector<double> out = fit::sdf_sample(pts.fetch(), &h);
		return Variant(PackedFloat64Array(out));
	});
}

Variant fit_probe(String what_s) {
	return guarded("fit_probe", [&] {
		const std::string what = what_s.utf8();
		if (what == "io")
			return text(fit::probe_io());
		if (what == "ldlt")
			return text(fit::probe_ldlt());
		if (what == "exceptions")
			return text(fit::probe_exceptions());
		if (what == "ldlt8k")
			return text(fit::probe_ldlt8k());
		if (what == "libm")
			return text(fit::probe_libm());
		if (what == "stl")
			return text(fit::probe_stl());
		if (what == "instret") {
			// Two reads around a known loop: the counter must advance by at
			// least the loop's instructions.
			const uint64_t a = instret();
			volatile uint64_t acc = 0;
			for (int i = 0; i < 100000; i++)
				acc += uint64_t(i);
			const uint64_t b = instret();
			return text(std::string(b - a >= 100000 ? "PASS" : "FAIL") + " instret: " + std::to_string(b - a) +
					" instructions over a 100000-iteration loop (at " + std::to_string(b) + " into this vmcall)");
		}
		if (what == "io_paths") {
			std::string s = "io_total " + std::to_string(fit::io_attempts());
			for (const std::string &p : fit::io_paths())
				s += "\n  " + p;
			return text(s);
		}
		if (what == "heap")
			return text(heap_text());
		return fail("fit_probe: want io | ldlt | ldlt8k | libm | stl | instret | exceptions | io_paths | heap");
	});
}

// --- cut 6g-C: the GPU Hessian's switches and its gate ----------------------------

Variant fit_set_gpu(int64_t mode) {
	return guarded("fit_set_gpu", [&] {
		if (mode < 0 || mode > 2)
			return fail("fit_set_gpu: mode 0 (CPU path), 1 (RenderingDevice) or 2 (cpp twin)");
		g_gpu_mode = int(mode);
		if (g_drv && g_drv->similarity_form())
			attach_gpu(*g_drv);
		return text("OK gpu mode " + std::to_string(g_gpu_mode));
	});
}

Variant fit_set_gpu_broad(int64_t mode) {
	return guarded("fit_set_gpu_broad", [&] {
		if (mode < 0 || mode > 2)
			return fail("fit_set_gpu_broad: mode 0 (CPU path), 1 (RenderingDevice) or 2 (RenderingDevice, audited by the CPU build)");
		g_bp_mode = int(mode);
		if (g_drv && g_drv->contact_form())
			attach_bp(*g_drv);
		return text("OK broad phase mode " + std::to_string(g_bp_mode));
	});
}

Variant fit_gpu_stats() {
	return guarded("fit_gpu_stats", [] {
		std::string s = gpu_text() + " | " + bp_text();
		if (g_sim)
			s += " | hinges " + std::to_string(g_sim->n_hinges) + " slots " + std::to_string(g_sim->n_slots) + " nnz " +
					std::to_string(g_sim->nnz()) + " session_data " + std::to_string(g_sim->bytes() / 1024) + " KiB";
		s += " | device " + g_gpu.device_name() + (g_gpu.is_open() ? " open" : " closed");
		return text(s);
	});
}

// Only from the thread that opened it (the worker): the device is bound to it.
Variant fit_gpu_close() {
	return guarded("fit_gpu_close", [] {
		const bool was = g_gpu.is_open() || g_bp.is_open();
		g_bp.close();
		g_gpu.close();
		return text(was ? "OK gpu closed" : "OK gpu was not open");
	});
}

// The C1 gate, on the worker thread after fit_begin (or after any phase):
// the blocks the kernels produce against the CPU double path at three
// solutions (the rest shape, a perturbed one with negative eigenvalues to
// project, the current one), psd off and on; the cpp twin against the
// device; the flipped-sign control; the assembled matrix's pattern and
// values; and the time each path takes. Every line is PASS or FAIL.
Variant fit_gpu_check() {
	return guarded("fit_gpu_check", [] {
		fit::FitDriver &d = driver();
		auto form = d.similarity_form();
		if (!form)
			return fail("fit_gpu_check: no similarity form");
		if (!g_sim) {
			g_sim = std::make_unique<fit_gpu::SimProblem>();
			g_sim->build(*form);
		}
		const fit_gpu::SimProblem &p = *g_sim;
		std::string err;
		if (!g_gpu.is_open() && !g_gpu.open(p, &err))
			return fail("fit_gpu_check: device: " + err);
		g_gpu.set_timestamps(true);
		std::string rep;
		int fails = 0;
		auto line = [&](bool pass, const std::string &s) {
			rep += std::string(pass ? "PASS " : "FAIL ") + s + "\n";
			if (!pass)
				fails++;
		};
		char b[512];
		std::snprintf(b, sizeof b, "problem: hinges %d slots %d n %d nnz %zu session_data %zu KiB device %s",
				p.n_hinges, p.n_slots, p.n, p.nnz(), p.bytes() / 1024, g_gpu.device_name().c_str());
		rep += std::string(b) + "\n";

		const Eigen::VectorXd cur = d.solution().col(0);
		Eigen::VectorXd rest = Eigen::VectorXd::Zero(cur.size());
		Eigen::VectorXd pert = cur;
		for (Eigen::Index i = 0; i < pert.size(); i++)
			pert(i) += 0.02 * std::sin(1.7 * double(i) + 0.3);
		const struct {
			const char *name;
			const Eigen::VectorXd *x;
		} arms[3] = { { "rest", &rest }, { "perturbed", &pert }, { "current", &cur } };
		const double kBlockTol = 1e-9, kTwinTol = 1e-12;
		for (const auto &arm : arms) {
			for (int psd = 0; psd < 2; psd++) {
				std::vector<double> ref;
				int64_t t0 = rdc::host_usec();
				form->hessian_blocks(*arm.x, psd != 0, ref);
				const int64_t cpu_us = rdc::host_usec() - t0;
				std::vector<fit_gpu::Df> rd, twin, pos;
				t0 = rdc::host_usec();
				const bool ok = g_gpu.blocks(p, *arm.x, psd != 0, 1.0f, rd);
				const int64_t rd_us = rdc::host_usec() - t0;
				if (!ok) {
					line(false, std::string(arm.name) + " psd=" + std::to_string(psd) + ": device: " + g_gpu.stats().last_error);
					continue;
				}
				t0 = rdc::host_usec();
				p.positions(*arm.x, pos);
				fit_gpu::SimTwin::blocks(p, pos, 1.0f, twin);
				fit_gpu::SimTwin::project_psd(p, psd != 0, twin);
				const int64_t twin_us = rdc::host_usec() - t0;
				const fit_gpu::BlockCompare c_rd = fit_gpu::compare_blocks(ref, rd, p.n_hinges);
				const fit_gpu::BlockCompare c_tw = fit_gpu::compare_blocks(twin, rd, p.n_hinges);
				const fit_gpu::BlockCompare c_tc = fit_gpu::compare_blocks(ref, twin, p.n_hinges);
				// How many blocks the projection changed on the CPU (the psd arm's evidence).
				int changed = 0;
				if (psd) {
					std::vector<double> raw;
					form->hessian_blocks(*arm.x, false, raw);
					for (int h = 0; h < p.n_hinges; h++)
						for (int e = 0; e < fit_gpu::kBlock; e++)
							if (raw[size_t(h) * fit_gpu::kBlock + e] != ref[size_t(h) * fit_gpu::kBlock + e]) {
								changed++;
								break;
							}
				}
				std::snprintf(b, sizeof b, "%s psd=%d rd vs cpu double: %s (tol %.0e); cpu %lld us, rd %lld us, twin %lld us%s",
						arm.name, psd, c_rd.text().c_str(), kBlockTol, (long long)cpu_us, (long long)rd_us, (long long)twin_us,
						psd ? (", projection changed " + std::to_string(changed) + " blocks").c_str() : "");
				line(c_rd.max_rel <= kBlockTol, b);
				std::snprintf(b, sizeof b, "%s psd=%d rd vs cpp twin: %s (tol %.0e, FMA noise); twin vs cpu double max_rel %.3e",
						arm.name, psd, c_tw.text().c_str(), kTwinTol, c_tc.max_rel);
				line(c_tw.max_rel <= kTwinTol, b);
			}
		}
		// The flipped-sign control: the same kernel with sign -1 must not pass.
		{
			std::vector<double> ref;
			std::vector<fit_gpu::Df> rd;
			form->hessian_blocks(pert, false, ref);
			const bool ok = g_gpu.blocks(p, pert, false, -1.0f, rd);
			const fit_gpu::BlockCompare c = ok ? fit_gpu::compare_blocks(ref, rd, p.n_hinges) : fit_gpu::BlockCompare{};
			std::snprintf(b, sizeof b, "control: sign-flipped kernel vs cpu double: %s -> the block check %s it",
					c.text().c_str(), c.max_rel > kBlockTol ? "FAILS" : "passes");
			line(ok && c.max_rel > kBlockTol, b);
		}
		// The assembled matrix: pattern and values, both paths, at the current solution.
		for (int psd = 0; psd < 2; psd++) {
			polyfem::StiffnessMatrix h_cpu, h_gpu, h_twin;
			form->set_hessian_hook(nullptr);
			form->set_project_to_psd(psd != 0);
			int64_t t0 = rdc::host_usec();
			form->second_derivative(cur, h_cpu);
			const int64_t cpu_us = rdc::host_usec() - t0;
			const int saved = g_gpu_mode;
			g_gpu_mode = 1;
			form->set_hessian_hook(&gpu_hook);
			t0 = rdc::host_usec();
			form->second_derivative(cur, h_gpu);
			const int64_t rd_us = rdc::host_usec() - t0;
			g_gpu_mode = 2;
			t0 = rdc::host_usec();
			form->second_derivative(cur, h_twin);
			const int64_t twin_us = rdc::host_usec() - t0;
			g_gpu_mode = saved;
			attach_gpu(d);
			const std::string pat = p.same_pattern(h_cpu);
			std::snprintf(b, sizeof b, "assembled psd=%d: pattern %s (nnz %lld)", psd, pat.empty() ? "identical" : pat.c_str(), (long long)h_cpu.nonZeros());
			line(pat.empty(), b);
			const double vr = fit_gpu::values_max_rel(h_cpu, h_gpu), vt = fit_gpu::values_max_rel(h_gpu, h_twin);
			std::snprintf(b, sizeof b, "assembled psd=%d: values rd vs cpu max|d|/max|cpu| %.3e (tol %.0e), twin vs rd %.3e; cpu %lld us, rd %lld us (%s), twin %lld us",
					psd, vr, kBlockTol, vt, (long long)cpu_us, (long long)rd_us, gpu_text().c_str(), (long long)twin_us);
			line(vr <= kBlockTol && vt <= kTwinTol, b);
		}
		g_gpu.set_timestamps(false);
		rep += "worker_syncs " + std::to_string(g_gpu.device().worker_syncs()) + " same_frame_syncs " +
				std::to_string(g_gpu.device().same_frame_syncs()) + " recoveries " + std::to_string(g_gpu.device().recoveries()) + "\n";
		rep += std::string(fails ? "FAIL " : "PASS ") + "fit_gpu_check: " + std::to_string(fails) + " failures";
		return text(rep);
	});
}

} // namespace

int main() {
	ADD_API_FUNCTION(fit_set_gpu, "String", "int mode", "Where the similarity Hessian is assembled: 0 CPU path, 1 RenderingDevice (default), 2 cpp twin");
	ADD_API_FUNCTION(fit_set_gpu_broad, "String", "int mode", "Where the CCD broad phase runs: 0 CPU path, 1 RenderingDevice (default), 2 RenderingDevice audited by the CPU build");
	ADD_API_FUNCTION(fit_gpu_check, "String", "", "Gate C1: kernels vs the CPU double path, the cpp twin, the flipped-sign control, pattern and timing (worker thread)");
	ADD_API_FUNCTION(fit_gpu_stats, "String", "", "GPU Hessian counters: assemblies, fallbacks, round-trip time, syncs");
	ADD_API_FUNCTION(fit_gpu_close, "String", "", "Free the GPU Hessian's device (from the worker thread that opened it)");
	ADD_API_FUNCTION(fit_reset, "String", "", "Drop the driver and every input");
	ADD_API_FUNCTION(fit_set_body, "String", "PackedFloat32Array v, PackedInt32Array f",
			"Target body (avatar) mesh, body space, xyz and triangle triples");
	ADD_API_FUNCTION(fit_set_skeletons, "String", "PackedFloat32Array src, PackedFloat32Array tgt, PackedInt32Array bones",
			"Source (garment) and target (body) skeleton joints, the same bones as index pairs");
	ADD_API_FUNCTION(fit_set_skin_weights, "String", "PackedFloat32Array w",
			"Optional body skin weights, n_bones x n_body_v column-major (empty: distance projection)");
	ADD_API_FUNCTION(fit_set_garment, "String", "PackedFloat32Array v, PackedInt32Array f, PackedInt32Array nofit",
			"Garment on the source body, its triangles and the vertex ids not fitted");
	ADD_API_FUNCTION(fit_set_config, "String", "String json", "cloth-fit setup JSON (keys naming files are dropped)");
	ADD_API_FUNCTION(fit_begin, "String", "", "Normalise, project, build the collision mesh, check the start state");
	ADD_API_FUNCTION(fit_step, "String", "", "Run the next phase (AL solve or reduced solve); DONE after the last");
	ADD_API_FUNCTION(fit_run_all, "String", "", "Run every remaining phase in this call");
	ADD_API_FUNCTION(fit_status, "String", "", "Phase, Newton iterations, energy, io_attempts, heap");
	ADD_API_FUNCTION(fit_result_vertices, "Variant", "", "Current garment, body space, PackedFloat32Array");
	ADD_API_FUNCTION(fit_result_vertices_f64, "Variant", "", "Current garment, solve frame, PackedFloat64Array");
	ADD_API_FUNCTION(fit_preview, "Variant", "int k", "Set the preview interval (k > 0); newest snapshot, body space");
	ADD_API_FUNCTION(fit_check_intersections, "String", "PackedFloat32Array override",
			"Intersection check on the current state, or with the garment replaced (body space)");
	ADD_API_FUNCTION(fit_push_vertex, "Variant", "double dist",
			"Current garment (body space) with its closest vertex moved dist solve units into the avatar");
	ADD_API_FUNCTION(fit_sdf_dump, "Variant", "PackedFloat64Array pts",
			"SDF grid via the Lean kernel at solve-frame points: 10 doubles each, index space");
	ADD_API_FUNCTION(fit_probe, "String", "String what", "io | ldlt | ldlt8k | libm | stl | instret | exceptions | io_paths | heap");
	halt();
}
