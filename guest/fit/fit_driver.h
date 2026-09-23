// fit_driver: cloth-fit's garment retargeting (vendor/cloth-fit
// src/polyfem/garment/run_retarget.cpp:150-326, main.cpp) without file I/O,
// sliced into phases so a host can advance it one phase per vmcall.
//
// The same source compiles natively (tools/fit/fit_native.cpp, the flat
// control) and into fit.elf. Nothing in here opens a file: meshes, skeletons
// and the setup JSON arrive as arrays and a string, and the JSON rules come
// from specs embedded at build time (guest/fit/specs, tools/fit/embed_files.py).
//
// Phases: phase p = 2*substep + k.
//   k = 0: build the substep's penalty/Lagrangian/fit forms and the NL problem,
//          then the augmented-Lagrangian solve (ALSolver::solve_al).
//   k = 1: enable the fit form, then the reduced solve (ALSolver::solve_reduced).
// The NL problem, AL solver and forms persist between the two calls.
#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver {
class SimilarityForm;
class ContactForm;
}

namespace fit {

// Row-major arrays. Positions are xyz triples, faces vertex-index triples,
// bones vertex-index pairs.
struct FitInput {
	std::vector<double> avatar_v;          // target avatar
	std::vector<int> avatar_f;
	std::vector<double> garment_v;         // garment on the source body
	std::vector<int> garment_f;
	std::vector<double> skeleton_v;        // source skeleton
	std::vector<int> skeleton_b;
	std::vector<double> target_skeleton_v; // target skeleton (same bones)
	std::vector<int> target_skeleton_b;
	// Optional: n_bones x n_avatar_v, column-major like the upstream matrix
	// (entry (bone, vertex) at [bone + vertex * n_bones]). Empty = distance
	// projection only, as upstream when the file is missing.
	std::vector<double> target_avatar_skinning_weights;
	// Garment vertex ids that are not fitted (upstream no-fit.txt).
	std::vector<int> no_fit_vertices;
	// cloth-fit setup JSON without the *_path keys. Keys: see
	// vendor/cloth-fit/json-specs/input-spec.json. "input_coordinate_system"
	// defaults to Godot {"up":"+Y","forward":"+Z","handedness":"right"}, i.e.
	// no transform; "common" is refused (it names a file).
	std::string setup_json;
};

struct PhaseStats {
	int phase = -1;
	int substep = -1;
	int kind = -1;             // 0 = AL solve, 1 = reduced solve
	int post_steps = 0;        // post_step calls (one per minimize start + one per Newton iteration)
	int minimize_calls = 0;    // nonlinear solves inside the phase
	int newton_iterations = 0; // post_steps - minimize_calls
	double energy = 0;         // polysolve info()["energy"] after the last minimize
	double grad_norm = 0;      // polysolve info()["gradNorm"]
	std::string status;        // polysolve info()["status"]
	bool ok = false;
	std::string error;
};

// Intersection query result: ids[0] < 0 means intersection-free; otherwise
// ids = {edge v0, edge v1, face v0, face v1, face v2} in collision-mesh ids
// (as ipc::my_has_intersections returns them).
struct IntersectionReport {
	std::array<int, 5> ids{{-1, -1, -1, -1, -1}};
	bool intersects() const { return ids[0] >= 0; }
};

struct Normalisation {
	// Target avatar frame -> solve frame: x_solve = scale * x_avatar + center.
	double target_scale = 1;
	double center[3] = {0, 0, 0};
	// Source garment frame -> solve frame: x_solve = source_scale * (x - source_offset).
	double source_scale = 1;
	double source_offset[3] = {0, 0, 0};
	// Input convention -> Godot canonical (row-major 3x3, identity for Godot input).
	double to_canonical[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

class FitDriver {
public:
	FitDriver();
	~FitDriver();
	FitDriver(const FitDriver &) = delete;
	FitDriver &operator=(const FitDriver &) = delete;

	// Fills the GarmentSolver, normalises, projects the avatar to the
	// skeleton, builds the IPC collision mesh and checks the start state.
	// Returns false (and sets *error) on invalid input or an intersecting
	// start (the ids are then in start_intersections()).
	bool begin(const FitInput &in, std::string *error);

	int phase_count() const;  // 2 * incremental_steps, 0 before begin()
	int next_phase() const;   // phases done so far
	bool done() const { return began_ && next_phase_ >= phase_count(); }
	bool failed() const { return failed_; }

	// Runs the next phase. Returns false on failure (stats->error says why);
	// the driver is then failed and runs nothing more.
	bool step(PhaseStats *stats);

	// Current garment, row-major xyz, in the solve frame (what upstream
	// writes as step_garment_*.obj) or mapped back to the target avatar frame.
	void garment_solve_frame(std::vector<double> *v) const;
	void garment_avatar_frame(std::vector<double> *v) const;
	const std::vector<int> &garment_faces() const { return garment_f_; }
	int garment_vertex_count() const;

	const Normalisation &normalisation() const { return norm_; }
	// The SimilarityForm of this session (null before begin()): cut 6g-C
	// attaches its GPU Hessian hook to it (guest/fit/fit_gpu.cpp).
	std::shared_ptr<polyfem::solver::SimilarityForm> similarity_form() const;
	// The ContactForm of this session (null before begin()) and what its
	// broad phase needs: the collision mesh's vertices from
	// avatar_vertex_count() on are the garment's, and self_collision() is
	// cloth-fit's can_collide mode (contact.enabled).
	std::shared_ptr<polyfem::solver::ContactForm> contact_form() const;
	int avatar_vertex_count() const;
	bool self_collision() const;
	// The current solution in the forms' complete space (3 x collision
	// vertices, avatar then garment): what a form's hook sees as x.
	const Eigen::MatrixXd &solution() const;
	const IntersectionReport &start_intersections() const { return start_ids_; }

	// ipc::my_has_intersections on the current state: avatar at the target
	// (alpha of the last phase run) and the current garment.
	IntersectionReport check_intersections() const;
	// Same, with the garment positions replaced (solve frame, row-major).
	IntersectionReport check_intersections_with(const std::vector<double> &garment_v) const;

	// Preview ring: every preview_every-th post_step writes the garment
	// (solve frame) into one of kPreviewSlots slots. latest_preview returns
	// the sequence number of the newest snapshot (-1 if none) and copies it.
	static constexpr int kPreviewSlots = 4;
	void set_preview_every(int k) { preview_every_ = k < 1 ? 1 : k; }
	int64_t latest_preview(std::vector<double> *v) const;

	// The setup after rule injection (for logs).
	std::string resolved_args() const;

private:
	struct State;
	std::unique_ptr<State> s_;
	bool began_ = false;
	bool failed_ = false;
	int next_phase_ = 0;
	int preview_every_ = 1;
	std::vector<int> garment_f_;
	Normalisation norm_;
	IntersectionReport start_ids_;

	bool run_phase(int phase, PhaseStats *stats);
};

// The I/O-free init(): vendor/cloth-fit optimize.cpp:1371-1460 against the
// embedded rules. Throws std::runtime_error on invalid input.
std::string init_args(const std::string &setup_json);

// Embedded file lookup (generated by tools/fit/embed_files.py).
const char *embedded_file(const char *name, size_t *size);

} // namespace fit
