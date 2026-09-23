// fit_driver: see fit_driver.h. Line references are to vendor/cloth-fit at
// d2bd59a6: src/polyfem/main.cpp (the PolyFEM_bin the oracle runs),
// src/polyfem/garment/run_retarget.cpp:150-326 (the same loop as a library
// call) and src/polyfem/garment/optimize.cpp (init, the GarmentSolver).
#include "fit_driver.h"

#include <polyfem/garment/optimize.hpp>
#include <polyfem/garment/coordinate_system.hpp>
#include <polyfem/garment/GarmentNLProblem.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/garment_forms/GarmentForm.hpp>
#include <polyfem/solver/forms/garment_forms/GarmentALForm.hpp>
#include <polyfem/solver/forms/garment_forms/CurveConstraintForm.hpp>
#include <polyfem/solver/forms/garment_forms/CurveCenterTargetForm.hpp>
#include <polyfem/solver/forms/garment_forms/FitForm.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <ipc/ipc.hpp>
#include <ipc/utils/logger.hpp>

#include <igl/edges.h>

#include <jse/jse.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cmath>
#include <stdexcept>

using json = nlohmann::json;
using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::mesh;

namespace spdlog::level {
// optimize.cpp:58-76: the level names the setup JSON uses.
NLOHMANN_JSON_SERIALIZE_ENUM(
		spdlog::level::level_enum,
		{{spdlog::level::level_enum::trace, "trace"},
		 {spdlog::level::level_enum::debug, "debug"},
		 {spdlog::level::level_enum::info, "info"},
		 {spdlog::level::level_enum::warn, "warning"},
		 {spdlog::level::level_enum::err, "error"},
		 {spdlog::level::level_enum::critical, "critical"},
		 {spdlog::level::level_enum::off, "off"},
		 {spdlog::level::level_enum::trace, 0},
		 {spdlog::level::level_enum::debug, 1},
		 {spdlog::level::level_enum::info, 2},
		 {spdlog::level::level_enum::warn, 3},
		 {spdlog::level::level_enum::err, 3},
		 {spdlog::level::level_enum::critical, 4},
		 {spdlog::level::level_enum::off, 5}})
} // namespace spdlog::level

namespace fit {

namespace {

// optimize.cpp:364-386 (file-local there).
bool are_same_edges(const Eigen::MatrixXi &A, const Eigen::MatrixXi &B) {
	if (A.rows() != B.rows())
		return false;
	for (int i = 0; i < A.rows(); i++) {
		bool flag = false;
		for (int j = 0; j < B.rows(); j++) {
			if ((std::min(A(i, 0), A(i, 1)) == std::min(B(j, 0), B(j, 1))) && (std::max(A(i, 0), A(i, 1)) == std::max(B(j, 0), B(j, 1)))) {
				flag = true;
				break;
			}
		}
		if (!flag)
			return false;
	}
	return true;
}

Eigen::MatrixXd rows3(const std::vector<double> &a, const char *what) {
	if (a.size() % 3 != 0)
		throw std::runtime_error(std::string(what) + ": length is not a multiple of 3");
	const Eigen::Index n = Eigen::Index(a.size() / 3);
	Eigen::MatrixXd m(n, 3);
	for (Eigen::Index i = 0; i < n; i++)
		for (int j = 0; j < 3; j++)
			m(i, j) = a[size_t(i) * 3 + j];
	return m;
}

Eigen::MatrixXi rowsi(const std::vector<int> &a, int cols, int n_verts, const char *what) {
	if (a.size() % size_t(cols) != 0)
		throw std::runtime_error(std::string(what) + ": length is not a multiple of " + std::to_string(cols));
	const Eigen::Index n = Eigen::Index(a.size() / cols);
	Eigen::MatrixXi m(n, cols);
	for (Eigen::Index i = 0; i < n; i++)
		for (int j = 0; j < cols; j++) {
			const int v = a[size_t(i) * cols + j];
			if (v < 0 || v >= n_verts)
				throw std::runtime_error(std::string(what) + ": index " + std::to_string(v) + " out of range");
			m(i, j) = v;
		}
	return m;
}

const char *kInputSpec = "input-spec.resolved.json";

json init_json(const json &p_args_in) {
	// optimize.cpp:1371-1460 without the file system: apply_common_params
	// would read the "common" file, the rules come embedded with their jse
	// includes already resolved (tools/fit/resolve_specs.cpp), and there is no
	// output directory, spdlog flush thread or thread-count change.
	json args_in = p_args_in;
	if (args_in.contains("common"))
		throw std::runtime_error("setup: \"common\" names a file; merge it before passing the setup");
	// The spec requires the mesh paths; the meshes arrive as arrays instead,
	// so the paths are empty ("file" rules pass: jse skips file checks).
	for (const char *k : {"avatar_mesh_path", "garment_mesh_path", "no_fit_spec_path",
						  "source_skeleton_path", "target_skeleton_path", "avatar_skin_weights_path"})
		if (!args_in.contains(k))
			args_in[k] = "";

	json rules;
	jse::JSE jse;
	{
		jse.strict = false;
		jse.skip_file_check = true;
		size_t size = 0;
		const char *text = embedded_file(kInputSpec, &size);
		if (!text)
			throw std::runtime_error("no embedded input spec");
		rules = json::parse(text, text + size);
		polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
	}

	polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());

	if (args_in.contains("/solver/nonlinear"_json_pointer)) {
		if (args_in.contains("/solver/augmented_lagrangian/nonlinear"_json_pointer)) {
			json nonlinear = args_in["solver"]["nonlinear"];
			nonlinear.merge_patch(args_in["solver"]["augmented_lagrangian"]["nonlinear"]);
			args_in["solver"]["augmented_lagrangian"]["nonlinear"] = nonlinear;
		} else {
			args_in["solver"]["augmented_lagrangian"]["nonlinear"] = args_in["solver"]["nonlinear"];
		}
	}

	if (!jse.verify_json(args_in, rules))
		throw std::runtime_error("invalid setup json:\n" + jse.log2str());

	json args = jse.inject_defaults(args_in, rules);

	const spdlog::level::level_enum log_level = args["output"]["log"]["level"];
	spdlog::set_level(log_level);
	logger().set_level(log_level);
	ipc::logger().set_level(log_level);
	return args;
}

} // namespace

std::string init_args(const std::string &setup_json) {
	return init_json(json::parse(setup_json)).dump();
}

struct FitDriver::State {
	json args;
	GarmentSolver g;

	Eigen::MatrixXi collision_triangles;
	Eigen::MatrixXi collision_edges;
	Eigen::MatrixXd collision_vertices;
	ipc::CollisionMesh collision_mesh;
	int n_avatar_verts = 0;

	std::vector<Eigen::VectorXi> curves;
	Eigen::MatrixXd initial_garment_v;
	int total_steps = 0;

	std::vector<std::shared_ptr<Form>> persistent_forms;
	std::vector<std::shared_ptr<Form>> persistent_full_forms;
	std::shared_ptr<CurveSizeForm> curve_size_form;
	std::shared_ptr<SimilarityForm> similarity_form;
	std::shared_ptr<ContactForm> contact_form;
	bool self_collision = false;

	Eigen::MatrixXd sol;

	// Per substep, alive from phase 2s to phase 2s+1.
	std::shared_ptr<PointPenaltyForm> pen_form;
	std::shared_ptr<PointLagrangianForm> lagr_form;
	std::shared_ptr<FitForm<4>> fit_form;
	std::unique_ptr<GarmentNLProblem> nl_problem;
	using AL = ALSolver<GarmentNLProblem, PointLagrangianForm, PointPenaltyForm>;
	std::unique_ptr<AL> al_solver;
	std::shared_ptr<polysolve::nonlinear::Solver> nl_solver;
	double alpha = 0;

	// Counters for the phase in progress.
	int post_steps = 0;
	int minimize_calls = 0;

	// Preview ring.
	int64_t preview_seq = -1;
	int64_t post_step_total = 0;
	std::array<std::vector<double>, FitDriver::kPreviewSlots> ring;
	std::array<int64_t, FitDriver::kPreviewSlots> ring_seq{{-1, -1, -1, -1}};

	Eigen::MatrixXd garment_at(const Eigen::VectorXd &x) const {
		// GarmentSolver::save_result (optimize.cpp:442-444, 490).
		const Eigen::VectorXd full = nl_problem ? nl_problem->reduced_to_full(x) : Eigen::VectorXd(x);
		const Eigen::MatrixXd garment0 = collision_vertices.bottomRows(g.n_garment_vertices());
		return utils::unflatten(full.tail(full.size() - 1), 3) + garment0;
	}
};

FitDriver::FitDriver() = default;
FitDriver::~FitDriver() = default;

int FitDriver::phase_count() const { return began_ ? 2 * s_->total_steps : 0; }
int FitDriver::next_phase() const { return next_phase_; }
int FitDriver::garment_vertex_count() const { return began_ ? s_->g.n_garment_vertices() : 0; }

std::string FitDriver::resolved_args() const { return s_ ? s_->args.dump() : std::string(); }

bool FitDriver::begin(const FitInput &in, std::string *error) {
	auto fail = [&](const std::string &e) {
		if (error)
			*error = e;
		failed_ = true;
		return false;
	};
	if (began_ || s_)
		return fail("begin: already begun");
	s_ = std::make_unique<State>();
	State &s = *s_;
	try {
		s.args = init_json(json::parse(in.setup_json));
		json &args = s.args;
		GarmentSolver &g = s.g;

		// main.cpp:150-153.
		g.out_folder.clear(); // no debug writes (optimize.cpp:1057,1077 are guarded)
		g.out_format = args.value("/output/format"_json_pointer, std::string("obj"));
		g.shrink_blend = args.value("shrink_blend", 1e-2);
		g.shrink_normal_distance = args.value("shrink_normal_distance", 0.0);
		const bool self_collision = args["contact"]["enabled"];

		// GarmentSolver::read_meshes (optimize.cpp:1082-1186), from arrays.
		g.avatar_v = rows3(in.avatar_v, "avatar_v");
		g.avatar_f = rowsi(in.avatar_f, 3, int(g.avatar_v.rows()), "avatar_f");
		g.skeleton_v = rows3(in.skeleton_v, "skeleton_v");
		g.skeleton_b = rowsi(in.skeleton_b, 2, int(g.skeleton_v.rows()), "skeleton_b");
		g.target_skeleton_v = rows3(in.target_skeleton_v, "target_skeleton_v");
		g.target_skeleton_b = rowsi(in.target_skeleton_b, 2, int(g.target_skeleton_v.rows()), "target_skeleton_b");
		if (!are_same_edges(g.skeleton_b, g.target_skeleton_b))
			return fail("Inconsistent skeletons!");
		g.target_skeleton_b = g.skeleton_b;
		if (!in.target_avatar_skinning_weights.empty()) {
			const Eigen::Index nb = g.skeleton_v.rows(), nv = g.avatar_v.rows();
			if (Eigen::Index(in.target_avatar_skinning_weights.size()) != nb * nv)
				return fail("Inconsistent skin weights dimension with the number of vertices and bones!");
			g.target_avatar_skinning_weights = Eigen::Map<const Eigen::MatrixXd>(in.target_avatar_skinning_weights.data(), nb, nv);
		} else {
			g.target_avatar_skinning_weights.setZero(0, 0);
		}

		// GarmentSolver::load_garment_mesh (optimize.cpp:894-1017), from arrays.
		g.garment.v = rows3(in.garment_v, "garment_v");
		g.garment.f = rowsi(in.garment_f, 3, int(g.garment.v.rows()), "garment_f");
		g.not_fit_fids.clear();
		if (!in.no_fit_vertices.empty()) {
			Eigen::VectorXi vmask = Eigen::VectorXi::Zero(g.garment.v.rows());
			for (int v : in.no_fit_vertices) {
				if (v < 0 || v >= g.garment.v.rows())
					return fail("Vertex ID " + std::to_string(v) + " in no-fit list out of range!");
				vmask(v) = 1;
			}
			for (int i = 0; i < g.garment.f.rows(); i++)
				if (vmask(g.garment.f(i, 0)) && vmask(g.garment.f(i, 1)) && vmask(g.garment.f(i, 2)))
					g.not_fit_fids.push_back(i);
		}
		garment_f_ = in.garment_f;

		// run_retarget.cpp:124-145: input convention -> Godot canonical. The
		// default here is Godot itself (no transform); upstream main.cpp has no
		// such step, so a setup that names no convention solves as the oracle.
		{
			const garment::Convention input_cs = garment::parse_convention(
					args.contains("input_coordinate_system") ? args["input_coordinate_system"] : json::object(),
					garment::godot_canonical());
			const Eigen::Matrix3d M = garment::to_canonical(input_cs);
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					norm_.to_canonical[i * 3 + j] = M(i, j);
			if (!M.isIdentity(1e-9)) {
				garment::apply_to_rows(M, g.avatar_v);
				garment::apply_to_rows(M, g.garment.v);
				garment::apply_to_rows(M, g.skeleton_v);
				garment::apply_to_rows(M, g.target_skeleton_v);
				if (M.determinant() < 0) {
					garment::flip_winding(g.avatar_f);
					garment::flip_winding(g.garment.f);
					for (int i = 0; i < g.garment.f.rows(); i++)
						for (int j = 0; j < 3; j++)
							garment_f_[size_t(i) * 3 + j] = g.garment.f(i, j);
				}
			}
		}

		// The normalisation normalize_meshes() is about to apply
		// (optimize.cpp:1349-1369), recorded so the result can be mapped back
		// to the target avatar's frame. Same expressions, same order.
		{
			Eigen::MatrixXd sk = g.skeleton_v;
			const Eigen::RowVector3d center_offset = sk.colwise().sum() / sk.rows();
			sk.rowwise() -= center_offset;
			const double source_scaling = 2. / bbox_size(sk).maxCoeff();
			sk *= source_scaling;
			const double target_scaling = bbox_size(sk).maxCoeff() / bbox_size(g.target_skeleton_v).maxCoeff();
			const Eigen::Vector3d center = sk.colwise().sum() / sk.rows() - target_scaling * g.avatar_v.colwise().sum() / g.avatar_v.rows();
			norm_.target_scale = target_scaling;
			norm_.source_scale = source_scaling;
			for (int j = 0; j < 3; j++) {
				norm_.center[j] = center(j);
				norm_.source_offset[j] = center_offset(j);
			}
		}

		// main.cpp:156-157.
		g.normalize_meshes();
		g.project_avatar_to_skeleton();

		// main.cpp:166-187: the IPC collision mesh, avatar first.
		s.collision_triangles.resize(g.nc_avatar_f.rows() + g.n_garment_faces(), g.garment.f.cols());
		s.collision_triangles << g.nc_avatar_f, g.garment.f.array() + g.nc_avatar_v.rows();
		igl::edges(s.collision_triangles, s.collision_edges);

		s.collision_vertices.resize(g.nc_avatar_v.rows() + g.n_garment_vertices(), g.garment.v.cols());
		s.collision_vertices << g.skinny_avatar_v, g.garment.v;

		s.collision_mesh = ipc::CollisionMesh(s.collision_vertices, s.collision_edges, s.collision_triangles);
		s.n_avatar_verts = int(g.nc_avatar_v.rows());
		s.self_collision = self_collision;
		const int n_avatar_verts = s.n_avatar_verts;
		s.collision_mesh.can_collide = [n_avatar_verts, self_collision](size_t vi, size_t vj) {
			if (self_collision)
				return vi >= size_t(n_avatar_verts) || vj >= size_t(n_avatar_verts);
			else
				return (vi >= size_t(n_avatar_verts) && vj < size_t(n_avatar_verts)) || (vi < size_t(n_avatar_verts) && vj >= size_t(n_avatar_verts));
		};

		// GarmentSolver::check_intersections (optimize.cpp:1019-1080) without
		// the debug OBJ writes: the ids are returned instead.
		start_ids_.ids = ipc::my_has_intersections(s.collision_mesh, s.collision_vertices, ipc::BroadPhaseMethod::BVH);
		if (start_ids_.intersects())
			return fail("Unable to solve, initial solution has intersections!");

		// main.cpp:212-275: boundary curves and the persistent forms.
		s.curves = boundary_curves(s.collision_triangles.bottomRows(g.n_garment_faces()));
		s.initial_garment_v = g.garment.v;
		s.total_steps = args["incremental_steps"];
		const auto &cv = s.collision_vertices;
		const Eigen::MatrixXi garment_tris = s.collision_triangles.bottomRows(g.n_garment_faces());

		auto similarity_form = std::make_shared<SimilarityForm>(cv, garment_tris);
		similarity_form->set_weight(args["similarity_penalty_weight"]);
		s.persistent_forms.push_back(similarity_form);
		s.similarity_form = similarity_form;

		if (args["curvature_penalty_weight"] > 0) {
			auto curvature_form = std::make_shared<CurveCurvatureForm>(cv, s.curves);
			curvature_form->set_weight(args["curvature_penalty_weight"]);
			s.persistent_forms.push_back(curvature_form);
		}
		if (args["twist_penalty_weight"] > 0) {
			auto twist_form = std::make_shared<CurveTorsionForm>(cv, s.curves);
			twist_form->set_weight(args["twist_penalty_weight"]);
			s.persistent_forms.push_back(twist_form);
		}
		if (args["symmetry_weight"] > 0) {
			auto sym_form = std::make_shared<SymmetryForm>(cv, s.curves);
			sym_form->set_weight(args["symmetry_weight"]);
			if (sym_form->enabled())
				s.persistent_forms.push_back(sym_form);
		}
		if (args["curve_size_weight"] > 0) {
			s.curve_size_form = std::make_shared<CurveSizeForm>(cv, s.curves);
			s.curve_size_form->disable();
			s.curve_size_form->set_weight(args["curve_size_weight"]);
			s.persistent_forms.push_back(s.curve_size_form);
		}
		{
			const double dhat = args["contact"]["dhat"];
			auto contact_form = std::make_shared<ContactForm>(
					s.collision_mesh, dhat, 1, false, false, false, false,
					args["solver"]["contact"]["CCD"]["broad_phase"],
					args["solver"]["contact"]["CCD"]["tolerance"],
					args["solver"]["contact"]["CCD"]["max_iterations"]);
			contact_form->set_weight(1);
			contact_form->set_barrier_stiffness(args["solver"]["contact"]["barrier_stiffness"]);
			contact_form->save_ccd_debug_meshes = false;
			s.persistent_forms.push_back(contact_form);
			s.contact_form = contact_form;
		}
		{
			const auto tmp_curves = boundary_curves(g.garment.f);
			auto center_target_form = std::make_shared<CurveTargetForm>(
					s.initial_garment_v, tmp_curves, g.skeleton_v, g.target_skeleton_v, g.skeleton_b,
					args["is_skirt"], args["curve_center_target_automatic_bone_generation"]);
			center_target_form->set_weight(args["curve_center_target_weight"]);
			s.persistent_full_forms.push_back(center_target_form);
		}

		s.sol = Eigen::MatrixXd::Zero(1 + s.initial_garment_v.size(), 1);
	} catch (const std::exception &e) {
		return fail(std::string("begin: ") + e.what());
	}
	began_ = true;
	return true;
}

bool FitDriver::step(PhaseStats *stats) {
	PhaseStats local;
	PhaseStats &st = stats ? *stats : local;
	st = PhaseStats();
	if (!began_ || failed_) {
		st.error = failed_ ? "driver failed earlier" : "begin() not called";
		return false;
	}
	if (done()) {
		st.error = "all phases done";
		return false;
	}
	const int phase = next_phase_;
	st.phase = phase;
	st.substep = phase / 2;
	st.kind = phase % 2;
	bool ok = false;
	try {
		ok = run_phase(phase, &st);
	} catch (const std::exception &e) {
		st.error = e.what();
		ok = false;
	}
	State &s = *s_;
	st.post_steps = s.post_steps;
	st.minimize_calls = s.minimize_calls;
	st.newton_iterations = s.post_steps - s.minimize_calls;
	if (s.nl_solver) {
		const json &info = s.nl_solver->info();
		if (info.contains("energy") && info["energy"].is_number())
			st.energy = info["energy"];
		if (info.contains("gradNorm") && info["gradNorm"].is_number())
			st.grad_norm = info["gradNorm"];
		if (info.contains("status"))
			st.status = info["status"].is_string() ? info["status"].get<std::string>() : info["status"].dump();
	}
	st.ok = ok;
	if (!ok)
		failed_ = true;
	++next_phase_;
	return ok;
}

bool FitDriver::run_phase(int phase, PhaseStats *st) {
	State &s = *s_;
	json &args = s.args;
	GarmentSolver &g = s.g;
	const int substep = phase / 2;
	s.post_steps = 0;
	s.minimize_calls = 0;

	if (phase % 2 == 0) {
		// main.cpp:280-345 (run_retarget.cpp:277-315): the substep's forms,
		// the NL problem, and the augmented-Lagrangian solve.
		const double next_alpha = (substep + 1) / (double)s.total_steps;
		s.alpha = next_alpha;
		logger().info("Start substep {} out of {}", substep + 1, s.total_steps);

		const Eigen::MatrixXd next_avatar_v = (g.nc_avatar_v - g.skinny_avatar_v) * next_alpha + g.skinny_avatar_v;

		// Upstream's per-substep objects are loop locals, gone before the next
		// substep builds its own; drop the previous substep's here so two SDFs
		// are never alive at once.
		s.al_solver.reset();
		s.nl_problem.reset();
		s.nl_solver.reset();
		s.fit_form.reset();
		s.pen_form.reset();
		s.lagr_form.reset();

		std::vector<std::shared_ptr<Form>> forms = s.persistent_forms;
		{
			std::vector<int> indices(g.nc_avatar_v.size());
			for (int i = 0; i < int(indices.size()); i++)
				indices[i] = i;
			s.pen_form = std::make_shared<PointPenaltyForm>(utils::flatten(next_avatar_v - g.skinny_avatar_v), indices);
			forms.push_back(s.pen_form);

			s.lagr_form = std::make_shared<PointLagrangianForm>(utils::flatten(next_avatar_v - g.skinny_avatar_v), indices);
			forms.push_back(s.lagr_form);

			s.fit_form = std::make_shared<FitForm<4>>(s.collision_vertices, s.collision_triangles.bottomRows(g.n_garment_faces()),
													  g.avatar_v, g.avatar_f, args["voxel_size"], g.not_fit_fids
#ifdef FIT_FITFORM_OUT_DIR
													  // The OpenVDB FitForm: out_dir "" skips its debug OBJ exports.
													  , std::string()
#endif
			);
			s.fit_form->disable();
			s.fit_form->set_weight(args["fit_weight"]);
			forms.push_back(s.fit_form);

			if (args["curve_size_weight"] > 0)
				s.curve_size_form->disable();
		}

		s.nl_problem = std::make_unique<GarmentNLProblem>(1 + s.initial_garment_v.size(), utils::flatten(g.nc_avatar_v - g.skinny_avatar_v), forms, s.persistent_full_forms);
		GarmentNLProblem &nl_problem = *s.nl_problem;
		nl_problem.set_target_value(next_alpha);

		nl_problem.line_search_begin(s.sol, s.sol);
		if (!std::isfinite(nl_problem.value(s.sol)) || !nl_problem.is_step_valid(s.sol, s.sol) || !nl_problem.is_step_collision_free(s.sol, s.sol)) {
			st->error = "Failed to apply boundary conditions!";
			return false;
		}

		s.nl_solver = polysolve::nonlinear::Solver::create(args["solver"]["augmented_lagrangian"]["nonlinear"], args["solver"]["linear"], 1., logger());

		double initial_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double scaling = args["solver"]["augmented_lagrangian"]["scaling"];
		const double max_weight = args["solver"]["augmented_lagrangian"]["max_weight"].get<double>();
		logger().debug("Set initial AL weight to {}", initial_weight);

		s.al_solver = std::make_unique<State::AL>(
				s.lagr_form, s.pen_form, initial_weight, scaling, max_weight,
				args["solver"]["augmented_lagrangian"]["eta"],
				args["solver"]["augmented_lagrangian"]["error_threshold"],
				[](const Eigen::VectorXd &) {});
		s.al_solver->post_subsolve = [&s](const double) { ++s.minimize_calls; };

		// main.cpp:327-331: upstream saves every stride-th iterate to disk;
		// here every preview_every_-th goes to the preview ring.
		const int every = preview_every_;
		nl_problem.post_step_call_back = [&s, every](const Eigen::VectorXd &x) {
			++s.post_steps;
			if (s.post_step_total++ % every == 0) {
				const int64_t seq = ++s.preview_seq;
				const int slot = int(seq % FitDriver::kPreviewSlots);
				const Eigen::MatrixXd gv = s.garment_at(x);
				std::vector<double> &out = s.ring[slot];
				out.resize(size_t(gv.rows()) * 3);
				for (Eigen::Index i = 0; i < gv.rows(); i++)
					for (int j = 0; j < 3; j++)
						out[size_t(i) * 3 + j] = gv(i, j);
				s.ring_seq[slot] = seq;
			}
		};

		s.al_solver->solve_al(s.nl_solver, nl_problem, s.sol);
		return true;
	}

	// main.cpp:335-341: fit form on, then the reduced solve.
	if (!s.nl_problem || !s.al_solver) {
		st->error = "reduced phase without its AL phase";
		return false;
	}
	s.fit_form->enable();
	if (args["curve_size_weight"] > 0 && substep == s.total_steps - 1)
		s.curve_size_form->enable();

	s.nl_solver = polysolve::nonlinear::Solver::create(args["solver"]["nonlinear"], args["solver"]["linear"], 1., logger());
	s.al_solver->solve_reduced(s.nl_solver, *s.nl_problem, s.sol);
	return true;
}

std::shared_ptr<polyfem::solver::SimilarityForm> FitDriver::similarity_form() const {
	return began_ ? s_->similarity_form : nullptr;
}

std::shared_ptr<polyfem::solver::ContactForm> FitDriver::contact_form() const {
	return began_ ? s_->contact_form : nullptr;
}

int FitDriver::avatar_vertex_count() const {
	return began_ ? s_->n_avatar_verts : 0;
}

bool FitDriver::self_collision() const {
	return began_ ? s_->self_collision : false;
}

const Eigen::MatrixXd &FitDriver::solution() const {
	static const Eigen::MatrixXd none;
	return began_ ? s_->sol : none;
}

void FitDriver::garment_solve_frame(std::vector<double> *v) const {
	v->clear();
	if (!began_)
		return;
	const State &s = *s_;
	// main.cpp:343: initial_garment_v + unflatten(sol.bottomRows(n), 3).
	const Eigen::MatrixXd gv = s.initial_garment_v + utils::unflatten(s.sol.bottomRows(s.initial_garment_v.size()), 3);
	v->resize(size_t(gv.rows()) * 3);
	for (Eigen::Index i = 0; i < gv.rows(); i++)
		for (int j = 0; j < 3; j++)
			(*v)[size_t(i) * 3 + j] = gv(i, j);
}

void FitDriver::garment_avatar_frame(std::vector<double> *v) const {
	garment_solve_frame(v);
	// x_solve = scale * (M x_in) + center  =>  x_in = M^T (x_solve - center) / scale.
	const double *M = norm_.to_canonical;
	for (size_t i = 0; i + 2 < v->size(); i += 3) {
		double c[3];
		for (int j = 0; j < 3; j++)
			c[j] = ((*v)[i + j] - norm_.center[j]) / norm_.target_scale;
		for (int j = 0; j < 3; j++)
			(*v)[i + j] = M[0 * 3 + j] * c[0] + M[1 * 3 + j] * c[1] + M[2 * 3 + j] * c[2];
	}
}

IntersectionReport FitDriver::check_intersections_with(const std::vector<double> &garment_v) const {
	IntersectionReport r;
	if (!began_)
		return r;
	const State &s = *s_;
	const int ng = s.g.n_garment_vertices();
	if (garment_v.size() != size_t(ng) * 3)
		throw std::runtime_error("check_intersections_with: wrong garment size");
	// The avatar where the current phase's alpha puts it (P * [alpha; 0]).
	Eigen::MatrixXd V = s.collision_vertices;
	const Eigen::MatrixXd avatar_disp = (s.g.nc_avatar_v - s.g.skinny_avatar_v) * s.alpha;
	V.topRows(s.n_avatar_verts) += avatar_disp;
	for (int i = 0; i < ng; i++)
		for (int j = 0; j < 3; j++)
			V(s.n_avatar_verts + i, j) = garment_v[size_t(i) * 3 + j];
	r.ids = ipc::my_has_intersections(s.collision_mesh, V, ipc::BroadPhaseMethod::BVH);
	return r;
}

IntersectionReport FitDriver::check_intersections() const {
	std::vector<double> gv;
	garment_solve_frame(&gv);
	return check_intersections_with(gv);
}

int64_t FitDriver::latest_preview(std::vector<double> *v) const {
	if (!s_ || s_->preview_seq < 0)
		return -1;
	const int slot = int(s_->preview_seq % kPreviewSlots);
	if (v)
		*v = s_->ring[slot];
	return s_->ring_seq[slot];
}

} // namespace fit
