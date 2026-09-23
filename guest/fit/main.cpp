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

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "fit_driver.h"
#include "fit_tools.h"

namespace {

std::unique_ptr<fit::FitDriver> g_drv;
fit::FitInput g_in;
bool g_have_body = false, g_have_skel = false, g_have_garment = false, g_have_config = false;
int g_io_at_begin = -1; // io_attempts when fit_begin started; -1 before
int g_newton_total = 0;
int g_preview_every = 1;
fit::PhaseStats g_last;

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
		const fit::Normalisation &nm = g_drv->normalisation();
		char b[320];
		std::snprintf(b, sizeof b, "OK begin: phases %d, target_scale %.17g center (%.17g, %.17g, %.17g), source_scale %.17g, io_attempts %d",
				g_drv->phase_count(), nm.target_scale, nm.center[0], nm.center[1], nm.center[2], nm.source_scale, io_in_window());
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
		const bool ok = d.step(&st);
		g_last = st;
		g_newton_total += st.newton_iterations;
		if (!ok)
			return fail("fit_step: " + phase_text(st) + ": " + st.error);
		return text("OK " + phase_text(st) + " io_attempts " + std::to_string(io_in_window()));
	});
}

Variant fit_run_all() {
	return guarded("fit_run_all", [] {
		fit::FitDriver &d = driver();
		std::string log;
		while (!d.done() && !d.failed()) {
			fit::PhaseStats st;
			const bool ok = d.step(&st);
			g_last = st;
			g_newton_total += st.newton_iterations;
			log += (ok ? "OK " : "FAIL ") + phase_text(st) + (ok ? "" : ": " + st.error) + "\n";
			if (!ok)
				return fail("fit_run_all:\n" + log);
		}
		return text(log + "DONE phases " + std::to_string(d.phase_count()) + " newton " + std::to_string(g_newton_total) +
				" io_attempts " + std::to_string(io_in_window()));
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
		std::snprintf(b, sizeof b, "phase %d/%d%s newton %d energy %.17g |grad| %.6g status %s io_attempts %d io_total %d %s",
				d.next_phase(), d.phase_count(), d.failed() ? " FAILED" : d.done() ? " done" : "", g_newton_total, g_last.energy,
				g_last.grad_norm, g_last.status.empty() ? "-" : g_last.status.c_str(), io_in_window(), fit::io_attempts(),
				heap_text().c_str());
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
		if (what == "io_paths") {
			std::string s = "io_total " + std::to_string(fit::io_attempts());
			for (const std::string &p : fit::io_paths())
				s += "\n  " + p;
			return text(s);
		}
		if (what == "heap")
			return text(heap_text());
		return fail("fit_probe: want io | ldlt | exceptions | io_paths | heap");
	});
}

} // namespace

int main() {
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
	ADD_API_FUNCTION(fit_sdf_dump, "Variant", "PackedFloat64Array pts",
			"SDF grid via the Lean kernel at solve-frame points: 10 doubles each, index space");
	ADD_API_FUNCTION(fit_probe, "String", "String what", "io | ldlt | exceptions | io_paths | heap");
	halt();
}
