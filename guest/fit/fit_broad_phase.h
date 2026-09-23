// fit_broad_phase -- the CCD broad phase on the fit worker's RenderingDevice
// (cut 6g-C, stage 2).
//
// ContactForm::line_search_begin asks for the candidate pairs of one line
// search: every edge-edge and face-vertex pair whose swept boxes (both ends
// of the search, inflated by dhat / 2) overlap and that the mesh's
// can_collide admits. ipc-toolkit answers with three SimpleBVH trees over
// the whole collision mesh (the CPU path, 23-43 % of a Newton iteration in
// the guest); this class answers with two Lean kernels (kernels/fit:
// swept_aabb, box_pair) in two round trips: boxes and a count per query,
// then the pairs at the host-scanned offsets. The queries are the
// garment-involved primitives only, and each pair comes out once, in a
// fixed order (Fit.SlangCodegen.BoxPair).
//
// The boxes are float, widened by a margin, so the GPU's set is a superset
// of the CPU's double set: the C2 gate audits every build of a phase
// against ipc::Candidates::build and counts misses (0) and extras.
#pragma once

#include "rd_compute.h"

#include <ipc/candidates/candidates.hpp>
#include <ipc/collision_mesh.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace fit_gpu {

struct BpStats {
	int64_t builds = 0;
	int64_t fallbacks = 0;
	int64_t total_us = 0;    // host time inside build()
	int64_t gpu_us = 0;      // of which the two round trips (upload, record, submit, sync, readback)
	int64_t convert_us = 0;  // of which scan + pairs -> candidates
	int64_t pairs = 0;       // pairs produced, all builds
	int64_t last_pairs = 0;
	int64_t last_us = 0;
	int64_t grown = 0;       // pair buffer reallocations
	// The audit (mode 2): ipc-toolkit's build on the CPU at the same inputs.
	int64_t audit_builds = 0;
	int64_t audit_cpu_us = 0;
	int64_t audit_cpu_pairs = 0;
	int64_t audit_missed = 0; // CPU pairs the GPU did not produce
	int64_t audit_extra = 0;  // GPU pairs the CPU did not produce
	std::string last_error;
};

class GpuBroadPhase {
public:
	// Uploads the mesh (edges, faces, the garment flags, the query list) to
	// the device the caller opened on this thread. Vertices from n_avatar
	// on are the garment's; self_collision is cloth-fit's can_collide mode.
	bool open(rdc::Device &dev, const ipc::CollisionMesh &mesh, int n_avatar, bool self_collision, std::string *err);
	bool is_open() const { return open_; }
	// The candidates of one line search into out (cleared first). false
	// (with stats().last_error) when the device refused; the caller then
	// falls back to the CPU path.
	bool build(const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, double inflation, ipc::Candidates &out);
	// Mode 2: after every build, ipc::Candidates::build with `method` on
	// the CPU, and the two sets compared into stats().
	void set_audit(bool on, ipc::BroadPhaseMethod method) {
		audit_ = on;
		method_ = method;
	}
	void close();
	const BpStats &stats() const { return stats_; }
	uint32_t queries() const { return n_q_; }

private:
	bool fail(const std::string &what);
	bool grow_pairs(size_t pairs);

	rdc::Device *dev_ = nullptr;
	const ipc::CollisionMesh *mesh_ = nullptr;
	bool open_ = false;
	bool audit_ = false;
	ipc::BroadPhaseMethod method_ = ipc::BroadPhaseMethod::BVH;
	BpStats stats_;
	uint32_t n_v_ = 0, n_e_ = 0, n_f_ = 0, n_q_ = 0;
	uint32_t n_q_edges_ = 0, n_q_verts_ = 0; // query segments: edges, vertices, faces
	bool self_ = false;
	std::vector<uint32_t> queries_;
	std::vector<float> x0_, x1_;
	std::vector<uint32_t> counts_, offsets_;
	std::vector<uint32_t> pairs_;
	size_t pair_cap_ = 0;

	::RID sh_aabb_, sh_pair_, pipe_aabb_, pipe_pair_;
	::RID b_x0_, b_x1_, b_edges_, b_faces_, b_garment_, b_boxes_, b_queries_, b_counts_, b_pairs_;
	::RID u_aabb_, u_count_, u_write_;
	::RID set_aabb_, set_count_, set_write_;
	float inflation_last_ = -1;
};

} // namespace fit_gpu
