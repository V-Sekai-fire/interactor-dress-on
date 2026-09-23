// fit_native: runs guest/fit/fit_driver natively, the flat control for fit.elf.
//
// Reads a cloth-fit setup (tools/native/foxgirl_oracle.json by default) and
// its meshes the way PolyFEM_bin does, rounds every position to float32 (as
// the Godot wire will: PackedVector3Array), strips the *_path keys and hands
// arrays + JSON to the driver, then runs the phases one call at a time and
// writes the final garment (solve frame, the frame upstream's
// step_garment_*.obj are in) as <out>/garment_final.f64 (row-major doubles)
// and <out>/garment_final.obj, plus <out>/phases.tsv.
//
//   fit_native [--setup S.json] [--root DIR] [--out DIR] [--phases N]
//              [--no-round] [--log LEVEL] [--set /json/pointer=VALUE]...
//              [--preview-every K] [--io-probe] [--control-push]
//   fit_native --compare A B [--voxel H]
//       A, B: .obj or .f64 garments with the same vertex order; prints the
//       per-vertex distance (max, mean, p99) in voxels and the vertex-sampled
//       Hausdorff distance (both directions, point to surface).
//   fit_native --probe ldlt8k|libm|stl|ldlt|exceptions [--reps N]
//       Gate 6.0: guest/fit/fit_probes.cpp, the code fit.elf's fit_probe runs.
//       Prints the probe's line and the wall time of each of N calls (min
//       first); the host times the guest's vmcall the same way.
//   --dump-inputs DIR: after reading (and rounding), write the arrays the
//       driver gets as raw little-endian files (avatar_v.f32, avatar_f.i32,
//       garment_v.f32, garment_f.i32, skeleton_v.f32, target_skeleton_v.f32,
//       skeleton_b.i32, no_fit.i32) so the Godot wire can be held to them bit
//       for bit (gate_fit.gd's wire check).
//
// --io-probe: the binary is linked with --wrap for fopen/_wfopen/_open/
// _wopen/_sopen_s/_wsopen_s/CreateFileA/CreateFileW, so every file open in
// the process is counted. The window from FitDriver::begin to the last phase
// must count zero; the harness's own reads before it are the positive
// control that the wrapper sees opens at all.
#include "fit_driver.h"
#include "fit_probes.h"

#include <polyfem/garment/optimize.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJData.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#ifdef FIT_SDF_GRID
#include <polyfem/solver/forms/garment_forms/SdfGrid.hpp>
#endif

#include <igl/point_mesh_squared_distance.h>
#include <nlohmann/json.hpp>
#include <tbb/global_control.h>

#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// I/O probe.
namespace {
std::atomic<int> g_io_attempts{0};
std::mutex g_io_mu;
std::vector<std::string> g_io_paths;

void io_note(const std::string &what) {
	g_io_attempts.fetch_add(1);
	std::lock_guard<std::mutex> lk(g_io_mu);
	if (g_io_paths.size() < 64)
		g_io_paths.push_back(what);
}

std::string narrow(const wchar_t *w) {
	if (!w)
		return "(null)";
	std::string s;
	for (; *w; ++w)
		s.push_back(*w < 128 ? char(*w) : '?');
	return s;
}
} // namespace

extern "C" {
FILE *__real_fopen(const char *, const char *);
FILE *__wrap_fopen(const char *p, const char *m) {
	io_note(std::string("fopen ") + (p ? p : "(null)"));
	return __real_fopen(p, m);
}
FILE *__real__wfopen(const wchar_t *, const wchar_t *);
FILE *__wrap__wfopen(const wchar_t *p, const wchar_t *m) {
	io_note("_wfopen " + narrow(p));
	return __real__wfopen(p, m);
}
int __real__open(const char *, int, ...);
int __wrap__open(const char *p, int flags, ...) {
	io_note(std::string("_open ") + (p ? p : "(null)"));
	return __real__open(p, flags, 0666);
}
int __real__wopen(const wchar_t *, int, ...);
int __wrap__wopen(const wchar_t *p, int flags, ...) {
	io_note("_wopen " + narrow(p));
	return __real__wopen(p, flags, 0666);
}
errno_t __real__sopen_s(int *, const char *, int, int, int);
errno_t __wrap__sopen_s(int *fd, const char *p, int o, int sh, int pm) {
	io_note(std::string("_sopen_s ") + (p ? p : "(null)"));
	return __real__sopen_s(fd, p, o, sh, pm);
}
errno_t __real__wsopen_s(int *, const wchar_t *, int, int, int);
errno_t __wrap__wsopen_s(int *fd, const wchar_t *p, int o, int sh, int pm) {
	io_note("_wsopen_s " + narrow(p));
	return __real__wsopen_s(fd, p, o, sh, pm);
}
HANDLE WINAPI __real_CreateFileA(LPCSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
HANDLE WINAPI __wrap_CreateFileA(LPCSTR a, DWORD b, DWORD c, LPSECURITY_ATTRIBUTES d, DWORD e, DWORD f, HANDLE g) {
	io_note(std::string("CreateFileA ") + (a ? a : "(null)"));
	return __real_CreateFileA(a, b, c, d, e, f, g);
}
HANDLE WINAPI __real_CreateFileW(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
HANDLE WINAPI __wrap_CreateFileW(LPCWSTR a, DWORD b, DWORD c, LPSECURITY_ATTRIBUTES d, DWORD e, DWORD f, HANDLE g) {
	io_note("CreateFileW " + narrow(a));
	return __real_CreateFileW(a, b, c, d, e, f, g);
}
}

// ---------------------------------------------------------------------------
namespace {

double cpu_seconds() {
	FILETIME c, e, k, u;
	GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
	auto t = [](const FILETIME &f) { return (double(f.dwHighDateTime) * 4294967296.0 + double(f.dwLowDateTime)) * 1e-7; };
	return t(k) + t(u);
}

double peak_ws_mb() {
	PROCESS_MEMORY_COUNTERS pmc{};
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	return double(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
}

double now_s() {
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// read_meshes / load_garment_mesh's reading and triangulation
// (optimize.cpp:1088-1127, 898-937).
bool read_tri_mesh(const std::string &path, std::vector<double> *v, std::vector<int> *f) {
	polyfem::OBJData d;
	if (!polyfem::read_mesh_with_groups(path, d))
		return false;
	v->clear();
	f->clear();
	for (const auto &p : d.V)
		for (int j = 0; j < 3; j++)
			v->push_back(p[j]);
	for (const auto &face : d.F) {
		if (face.size() == 3)
			f->insert(f->end(), face.begin(), face.end());
		else if (face.size() > 3)
			for (size_t i = 1; i < face.size() - 1; ++i)
				f->insert(f->end(), {face[0], face[i], face[i + 1]});
	}
	return true;
}

void to_vec(const Eigen::MatrixXd &m, std::vector<double> *v) {
	v->resize(size_t(m.rows()) * 3);
	for (Eigen::Index i = 0; i < m.rows(); i++)
		for (int j = 0; j < 3; j++)
			(*v)[size_t(i) * 3 + j] = m(i, j);
}

void to_vec(const Eigen::MatrixXi &m, std::vector<int> *v) {
	v->resize(size_t(m.rows()) * m.cols());
	for (Eigen::Index i = 0; i < m.rows(); i++)
		for (Eigen::Index j = 0; j < m.cols(); j++)
			(*v)[size_t(i) * m.cols() + j] = m(i, j);
}

void round_f32(std::vector<double> *v) {
	for (double &x : *v)
		x = double(float(x));
}

std::string join(const std::string &root, const std::string &p) {
	if (p.empty() || root.empty() || p[0] == '/' || (p.size() > 1 && p[1] == ':'))
		return p;
	return root + "/" + p;
}

bool load_garment(const std::string &path, Eigen::MatrixXd *V, Eigen::MatrixXi *F) {
	std::vector<double> v;
	std::vector<int> f;
	if (path.size() > 4 && path.substr(path.size() - 4) == ".f64") {
		std::ifstream in(path, std::ios::binary);
		if (!in)
			return false;
		in.seekg(0, std::ios::end);
		const size_t n = size_t(in.tellg()) / sizeof(double);
		in.seekg(0);
		v.resize(n);
		in.read(reinterpret_cast<char *>(v.data()), std::streamsize(n * sizeof(double)));
	} else {
		std::ifstream in(path);
		if (!in)
			return false;
		std::string line;
		while (std::getline(in, line)) {
			std::istringstream ss(line);
			std::string tag;
			ss >> tag;
			if (tag == "v") {
				double x, y, z;
				ss >> x >> y >> z;
				v.insert(v.end(), {x, y, z});
			} else if (tag == "f") {
				std::vector<int> ids;
				std::string tok;
				while (ss >> tok)
					ids.push_back(std::stoi(tok.substr(0, tok.find('/'))) - 1);
				for (size_t i = 1; i + 1 < ids.size(); i++)
					f.insert(f.end(), {ids[0], ids[i], ids[i + 1]});
			}
		}
	}
	V->resize(Eigen::Index(v.size() / 3), 3);
	for (Eigen::Index i = 0; i < V->rows(); i++)
		for (int j = 0; j < 3; j++)
			(*V)(i, j) = v[size_t(i) * 3 + j];
	F->resize(Eigen::Index(f.size() / 3), 3);
	for (Eigen::Index i = 0; i < F->rows(); i++)
		for (int j = 0; j < 3; j++)
			(*F)(i, j) = f[size_t(i) * 3 + j];
	return true;
}

int compare(const std::string &a, const std::string &b, double h) {
	Eigen::MatrixXd VA, VB;
	Eigen::MatrixXi FA, FB;
	if (!load_garment(a, &VA, &FA) || !load_garment(b, &VB, &FB)) {
		std::fprintf(stderr, "compare: cannot read %s or %s\n", a.c_str(), b.c_str());
		return 2;
	}
	if (VA.rows() != VB.rows()) {
		std::fprintf(stderr, "compare: %lld vs %lld vertices\n", (long long)VA.rows(), (long long)VB.rows());
		return 2;
	}
	std::vector<double> d(size_t(VA.rows()));
	double mx = 0, sum = 0;
	Eigen::Index arg = 0;
	for (Eigen::Index i = 0; i < VA.rows(); i++) {
		d[size_t(i)] = (VA.row(i) - VB.row(i)).norm();
		sum += d[size_t(i)];
		if (d[size_t(i)] > mx) {
			mx = d[size_t(i)];
			arg = i;
		}
	}
	std::vector<double> s = d;
	std::sort(s.begin(), s.end());
	const double p99 = s[size_t(std::floor(0.99 * double(s.size() - 1)))];
	double haus = -1, hab = -1, hba = -1;
	const Eigen::MatrixXi &F = FA.rows() ? FA : FB;
	if (F.rows()) {
		Eigen::VectorXd sq;
		Eigen::VectorXi I;
		Eigen::MatrixXd C;
		igl::point_mesh_squared_distance(VA, VB, F, sq, I, C);
		hab = std::sqrt(sq.maxCoeff());
		igl::point_mesh_squared_distance(VB, VA, F, sq, I, C);
		hba = std::sqrt(sq.maxCoeff());
		haus = std::max(hab, hba);
	}
	std::printf("compare %s vs %s: %lld verts, voxel %g\n", a.c_str(), b.c_str(), (long long)VA.rows(), h);
	std::printf("  per-vertex: max %.6g voxels (vertex %lld), mean %.6g, p99 %.6g  [max %.6e m]\n",
				mx / h, (long long)arg, sum / double(VA.rows()) / h, p99 / h, mx);
	std::printf("  hausdorff (vertex-sampled, point to surface): %.6g voxels (A->B %.6g, B->A %.6g)\n",
				haus / h, hab / h, hba / h);
	return 0;
}

bool write_outputs(const std::string &dir, const std::vector<double> &v, const std::vector<int> &f) {
	{
		std::ofstream o(dir + "/garment_final.f64", std::ios::binary);
		o.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(double)));
		if (!o)
			return false;
	}
	std::ofstream o(dir + "/garment_final.obj");
	char buf[128];
	for (size_t i = 0; i + 2 < v.size(); i += 3) {
		std::snprintf(buf, sizeof buf, "v %.17g %.17g %.17g\n", v[i], v[i + 1], v[i + 2]);
		o << buf;
	}
	for (size_t i = 0; i + 2 < f.size(); i += 3)
		o << "f " << f[i] + 1 << " " << f[i + 1] + 1 << " " << f[i + 2] + 1 << "\n";
	return bool(o);
}

} // namespace

int main(int argc, char **argv) {
	std::string setup_path = "tools/native/foxgirl_oracle.json";
	std::string root = ".";
	std::string out_dir;
	std::string log_level;
	int max_phases = -1;
	int preview_every = 1;
	bool round = true, io_probe = false, control_push = false;
	double voxel = 0.01;
	std::vector<std::pair<std::string, std::string>> sets;
	std::string cmp_a, cmp_b;
	std::string probe, dump_dir;
	int reps = 5;

	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "%s needs a value\n", a.c_str());
				std::exit(2);
			}
			return argv[++i];
		};
		if (a == "--setup") setup_path = next();
		else if (a == "--root") root = next();
		else if (a == "--out") out_dir = next();
		else if (a == "--phases") max_phases = std::stoi(next());
		else if (a == "--no-round") round = false;
		else if (a == "--log") log_level = next();
		else if (a == "--preview-every") preview_every = std::stoi(next());
		else if (a == "--io-probe") io_probe = true;
		else if (a == "--control-push") control_push = true;
		else if (a == "--voxel") voxel = std::stod(next());
		else if (a == "--set") {
			const std::string kv = next();
			const size_t eq = kv.find('=');
			if (eq == std::string::npos) {
				std::fprintf(stderr, "--set wants /pointer=json\n");
				return 2;
			}
			sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
		} else if (a == "--probe") {
			probe = next();
		} else if (a == "--reps") {
			reps = std::stoi(next());
		} else if (a == "--dump-inputs") {
			dump_dir = next();
		} else if (a == "--compare") {
			cmp_a = next();
			cmp_b = next();
		} else {
			std::fprintf(stderr, "unknown argument %s\n", a.c_str());
			return 2;
		}
	}
	if (!cmp_a.empty())
		return compare(cmp_a, cmp_b, voxel);
	if (!probe.empty()) {
		std::vector<double> ms;
		std::string line;
		for (int r = 0; r < reps; r++) {
			const double t0 = now_s();
			const std::string l = probe == "ldlt8k" ? fit::probe_ldlt8k()
					: probe == "libm"               ? fit::probe_libm()
					: probe == "stl"                ? fit::probe_stl()
					: probe == "ldlt"               ? fit::probe_ldlt()
					: probe == "exceptions"         ? fit::probe_exceptions()
													: std::string("FAIL: unknown probe ") + probe;
			ms.push_back((now_s() - t0) * 1e3);
			if (r == 0)
				line = l;
			else if (l != line)
				line += " [NOT REPEATABLE: " + l + "]";
		}
		std::printf("%s\n", line.c_str());
		std::vector<double> s = ms;
		std::sort(s.begin(), s.end());
		std::printf("time_ms min %.3f median %.3f (reps %d:", s.front(), s[s.size() / 2], reps);
		for (double m : ms)
			std::printf(" %.3f", m);
		std::printf(")\n");
		return line.rfind("FAIL", 0) == 0 ? 1 : 0;
	}

	// The oracle of record runs one thread (tools/native/README.md). The
	// driver sets no thread count; ipc-toolkit's TBB is held to one here.
	tbb::global_control one_thread(tbb::global_control::max_allowed_parallelism, 1);

	const int io_before_inputs = g_io_attempts.load();
	json setup;
	{
		std::ifstream f(join(root, setup_path));
		if (!f) {
			std::fprintf(stderr, "cannot open %s\n", setup_path.c_str());
			return 2;
		}
		f >> setup;
	}
	for (const auto &kv : sets)
		setup[json::json_pointer(kv.first)] = json::parse(kv.second);
	if (!log_level.empty())
		setup["output"]["log"]["level"] = log_level;

	fit::FitInput in;
	{
		const std::string avatar = join(root, setup.value("avatar_mesh_path", ""));
		const std::string garment = join(root, setup.value("garment_mesh_path", ""));
		const std::string src_sk = join(root, setup.value("source_skeleton_path", ""));
		const std::string tgt_sk = join(root, setup.value("target_skeleton_path", ""));
		const std::string skin = setup.value("avatar_skin_weights_path", "");
		const std::string nofit = setup.value("no_fit_spec_path", "");
		if (!read_tri_mesh(avatar, &in.avatar_v, &in.avatar_f) || !read_tri_mesh(garment, &in.garment_v, &in.garment_f)) {
			std::fprintf(stderr, "cannot read %s / %s\n", avatar.c_str(), garment.c_str());
			return 2;
		}
		Eigen::MatrixXd V;
		Eigen::MatrixXi E;
		polyfem::mesh::read_edge_mesh(src_sk, V, E);
		to_vec(V, &in.skeleton_v);
		to_vec(E, &in.skeleton_b);
		polyfem::mesh::read_edge_mesh(tgt_sk, V, E);
		to_vec(V, &in.target_skeleton_v);
		to_vec(E, &in.target_skeleton_b);
		if (!skin.empty() && std::ifstream(join(root, skin)).good()) {
			Eigen::MatrixXd W;
			polyfem::io::read_matrix(join(root, skin), W);
			in.target_avatar_skinning_weights.assign(W.data(), W.data() + W.size());
		}
		if (!nofit.empty() && std::ifstream(join(root, nofit)).good()) {
			Eigen::MatrixXi ids;
			polyfem::io::read_matrix<int>(join(root, nofit), ids);
			in.no_fit_vertices.assign(ids.data(), ids.data() + ids.size());
		}
		if (round) {
			round_f32(&in.avatar_v);
			round_f32(&in.garment_v);
			round_f32(&in.skeleton_v);
			round_f32(&in.target_skeleton_v);
			round_f32(&in.target_avatar_skinning_weights);
		}
		if (!dump_dir.empty()) {
			if (!out_dir.empty())
				CreateDirectoryA(out_dir.c_str(), nullptr); // dump_dir may sit inside it
			CreateDirectoryA(dump_dir.c_str(), nullptr);
			auto dump_f = [&](const char *name, const std::vector<double> &v) {
				std::vector<float> f(v.begin(), v.end());
				std::ofstream o(dump_dir + "/" + name, std::ios::binary);
				o.write(reinterpret_cast<const char *>(f.data()), std::streamsize(f.size() * sizeof(float)));
			};
			auto dump_i = [&](const char *name, const std::vector<int> &v) {
				std::vector<int32_t> f(v.begin(), v.end());
				std::ofstream o(dump_dir + "/" + name, std::ios::binary);
				o.write(reinterpret_cast<const char *>(f.data()), std::streamsize(f.size() * sizeof(int32_t)));
			};
			dump_f("avatar_v.f32", in.avatar_v);
			dump_i("avatar_f.i32", in.avatar_f);
			dump_f("garment_v.f32", in.garment_v);
			dump_i("garment_f.i32", in.garment_f);
			dump_f("skeleton_v.f32", in.skeleton_v);
			dump_f("target_skeleton_v.f32", in.target_skeleton_v);
			dump_i("skeleton_b.i32", in.skeleton_b);
			dump_i("no_fit.i32", in.no_fit_vertices);
			std::printf("inputs dumped to %s\n", dump_dir.c_str());
		}
		for (const char *k : {"avatar_mesh_path", "garment_mesh_path", "source_skeleton_path", "target_skeleton_path",
							  "avatar_skin_weights_path", "no_fit_spec_path", "root_path"})
			setup.erase(k);
		in.setup_json = setup.dump();
	}
	const int io_inputs = g_io_attempts.load() - io_before_inputs;
	std::printf("inputs: avatar %zu v / %zu f, garment %zu v / %zu f, skeleton %zu v / %zu b, no-fit %zu, float32 %s, harness file opens %d\n",
				in.avatar_v.size() / 3, in.avatar_f.size() / 3, in.garment_v.size() / 3, in.garment_f.size() / 3,
				in.skeleton_v.size() / 3, in.skeleton_b.size() / 2, in.no_fit_vertices.size(), round ? "yes" : "no", io_inputs);
	std::fflush(stdout);

	// ---- the driver window: no file opens allowed from here to the last phase.
	const int io_window_start = g_io_attempts.load();
	fit::FitDriver drv;
	drv.set_preview_every(preview_every);
	std::string err;
	const double t0 = now_s(), c0 = cpu_seconds();
	const bool began = drv.begin(in, &err);
	const double t1 = now_s(), c1 = cpu_seconds();
	std::printf("begin: %s  wall %.2f s  cpu %.2f s\n", began ? "ok" : err.c_str(), t1 - t0, c1 - c0);
	if (!began) {
		const auto &ids = drv.start_intersections().ids;
		if (ids[0] >= 0)
			std::printf("start intersects: edge (%d,%d) face (%d,%d,%d)\n", ids[0], ids[1], ids[2], ids[3], ids[4]);
		return 1;
	}
	const fit::Normalisation &nm = drv.normalisation();
	std::printf("normalisation: target_scale %.17g center (%.17g, %.17g, %.17g) source_scale %.17g\n",
				nm.target_scale, nm.center[0], nm.center[1], nm.center[2], nm.source_scale);

	struct Row {
		fit::PhaseStats st;
		double wall, cpu;
	};
	std::vector<Row> rows;
	int total_newton = 0, total_post = 0;
	const int n_phases = max_phases < 0 ? drv.phase_count() : std::min(max_phases, drv.phase_count());
	bool ok = true;
	for (int p = 0; p < n_phases; p++) {
		fit::PhaseStats st;
		const double w0 = now_s(), cc0 = cpu_seconds();
		ok = drv.step(&st);
		const double w1 = now_s(), cc1 = cpu_seconds();
		rows.push_back({st, w1 - w0, cc1 - cc0});
		total_newton += st.newton_iterations;
		total_post += st.post_steps;
		std::printf("phase %d (substep %d, %s): %s  newton %d  minimize %d  post_steps %d  energy %.17g  |grad| %.6g  status %s  wall %.2f s  cpu %.2f s\n",
					st.phase, st.substep, st.kind == 0 ? "AL" : "reduced", ok ? "ok" : st.error.c_str(),
					st.newton_iterations, st.minimize_calls, st.post_steps, st.energy, st.grad_norm, st.status.c_str(),
					w1 - w0, cc1 - cc0);
		std::fflush(stdout);
		if (!ok)
			break;
	}
	const int io_window = g_io_attempts.load() - io_window_start;
	// ---- end of the driver window.

	std::vector<double> gv;
	drv.garment_solve_frame(&gv);
	const fit::IntersectionReport rep = drv.check_intersections();
	std::printf("final: phases %zu/%d  newton %d  post_steps %d  intersections %s",
				rows.size(), drv.phase_count(), total_newton, total_post, rep.intersects() ? "INTERSECTS" : "none");
	if (rep.intersects())
		std::printf(" edge (%d,%d) face (%d,%d,%d)", rep.ids[0], rep.ids[1], rep.ids[2], rep.ids[3], rep.ids[4]);
	std::printf("\n");
	std::vector<double> pv;
	const int64_t seq = drv.latest_preview(&pv);
	double pdiff = -1;
	if (seq >= 0 && pv.size() == gv.size()) {
		pdiff = 0;
		for (size_t i = 0; i < gv.size(); i++)
			pdiff = std::max(pdiff, std::abs(pv[i] - gv[i]));
	}
	std::printf("preview ring: latest seq %lld, max |preview - final| %.3g\n", (long long)seq, pdiff);
	std::printf("time: begin+phases wall %.2f s, cpu %.2f s; process cpu %.2f s; peak working set %.1f MB\n",
				now_s() - t0, cpu_seconds() - c0, cpu_seconds(), peak_ws_mb());

#ifdef FIT_SDF_GRID
	if (const auto grid = polyfem::solver::SdfGrid::current()) {
		const auto gs = grid->stats();
		std::printf("sdf grid: voxel %g  bricks %zu (%zu voxels)  %.2f MB  distance queries %zu  winding queries %zu  fill %.2f s\n",
					grid->voxel_size(), gs.bricks, gs.bricks * std::size_t(polyfem::solver::SdfGrid::kBrickVoxels),
					gs.bytes / 1048576.0, gs.distance_queries, gs.winding_queries, gs.fill_seconds);
	}
#endif

	if (control_push) {
		// Control for the intersection check: one garment vertex pushed into
		// the avatar must be reported.
		// The avatar's centroid in the solve frame is inside the body; moving
		// the garment vertex nearest it 80% of the way there takes it (and
		// its edges) through the body surface.
		size_t best = 0;
		double cx = 0, cy = 0, cz = 0;
		const size_t na = in.avatar_v.size() / 3;
		for (size_t i = 0; i < na; i++) {
			cx += in.avatar_v[3 * i];
			cy += in.avatar_v[3 * i + 1];
			cz += in.avatar_v[3 * i + 2];
		}
		cx = cx / double(na) * nm.target_scale + nm.center[0];
		cy = cy / double(na) * nm.target_scale + nm.center[1];
		cz = cz / double(na) * nm.target_scale + nm.center[2];
		double bestd = 1e300;
		for (size_t i = 0; i + 2 < gv.size(); i += 3) {
			const double d = std::hypot(gv[i] - cx, gv[i + 1] - cy, gv[i + 2] - cz);
			if (d < bestd) {
				bestd = d;
				best = i;
			}
		}
		std::vector<double> pushed = gv;
		const double t = 0.8;
		for (int j = 0; j < 3; j++) {
			const double c = j == 0 ? cx : j == 1 ? cy : cz;
			pushed[best + j] = gv[best + j] + t * (c - gv[best + j]);
		}
		const fit::IntersectionReport pr = drv.check_intersections_with(pushed);
		std::printf("control-push: vertex %zu moved %.4f toward the avatar centroid -> %s\n",
					best / 3, t * bestd, pr.intersects() ? "INTERSECTS (control passes)" : "none (control FAILS)");
	}

	if (io_probe) {
		std::printf("io-probe: harness opens before the driver %d (positive control), io_attempts inside the driver %d\n",
					io_inputs, io_window);
		std::lock_guard<std::mutex> lk(g_io_mu);
		for (const auto &p : g_io_paths)
			std::printf("  io: %s\n", p.c_str());
	}

	if (!out_dir.empty()) {
		CreateDirectoryA(out_dir.c_str(), nullptr);
		if (!write_outputs(out_dir, gv, drv.garment_faces()))
			std::fprintf(stderr, "cannot write %s\n", out_dir.c_str());
		std::ofstream t(out_dir + "/phases.tsv");
		t << "phase\tsubstep\tkind\tok\tnewton\tminimize\tpost_steps\tenergy\tgrad_norm\tstatus\twall_s\tcpu_s\n";
		char buf[512];
		for (const Row &r : rows) {
			std::snprintf(buf, sizeof buf, "%d\t%d\t%s\t%d\t%d\t%d\t%d\t%.17g\t%.6g\t%s\t%.3f\t%.3f\n", r.st.phase, r.st.substep,
						  r.st.kind == 0 ? "AL" : "reduced", r.st.ok ? 1 : 0, r.st.newton_iterations, r.st.minimize_calls,
						  r.st.post_steps, r.st.energy, r.st.grad_norm, r.st.status.c_str(), r.wall, r.cpu);
			t << buf;
		}
	}
	if (io_probe && io_window != 0)
		return 3;
	return ok ? 0 : 1;
}
