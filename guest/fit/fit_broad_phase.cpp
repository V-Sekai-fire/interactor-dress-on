// fit_broad_phase -- the CCD broad phase on the worker's RenderingDevice. See fit_broad_phase.h.
#include "fit_broad_phase.h"

#include "rd_enums.h"

#include "fit_kernels.inc"

#include <ipc/candidates/edge_edge.hpp>
#include <ipc/candidates/face_vertex.hpp>

#include <algorithm>
#include <unordered_set>

namespace fit_gpu {

namespace {

struct AabbParams {
	uint32_t n_v, n_e, n_f;
	float inflation, margin;
	uint32_t pad[3];
};
struct PairParams {
	uint32_t n_queries, n_v, n_e, n_f, self_collision, write;
	uint32_t pad[2];
};

// 2^-20: about 16 float ulps at the coordinates' scale (Fit.SlangCodegen.SweptAabb).
constexpr float kMargin = 9.5367431640625e-07f;
constexpr size_t kPairsInitial = size_t(1) << 18;

const uint8_t *kernel(const char *name, size_t &bytes) {
	const fit_kernels::Entry *e = fit_kernels::find(name);
	if (!e) {
		bytes = 0;
		return nullptr;
	}
	bytes = e->size;
	return e->bytes;
}

inline uint64_t key(uint32_t a, uint32_t b) {
	return (uint64_t(a) << 32) | uint64_t(b);
}

} // namespace

bool GpuBroadPhase::fail(const std::string &what) {
	stats_.last_error = what + (dev_ && !dev_->error().empty() ? " (" + dev_->error() + ")" : "");
	return false;
}

bool GpuBroadPhase::open(rdc::Device &dev, const ipc::CollisionMesh &mesh, int n_avatar, bool self_collision, std::string *err) {
	if (open_)
		return true;
	auto bail = [&](const std::string &what) {
		if (err)
			*err = what + (dev.error().empty() ? "" : ": " + dev.error());
		close();
		return false;
	};
	if (!dev.ok())
		return bail("device not open");
	if (mesh.num_codim_vertices() != 0)
		return bail("codimensional vertices are not handled");
	dev_ = &dev;
	mesh_ = &mesh;
	self_ = self_collision;
	const Eigen::MatrixXi &E = mesh.edges();
	const Eigen::MatrixXi &F = mesh.faces();
	n_v_ = uint32_t(mesh.num_vertices());
	n_e_ = uint32_t(E.rows());
	n_f_ = uint32_t(F.rows());
	std::vector<uint32_t> edges(size_t(n_e_) * 2), faces(size_t(n_f_) * 3), garment(n_v_);
	for (uint32_t v = 0; v < n_v_; v++)
		garment[v] = v >= uint32_t(n_avatar) ? 1u : 0u;
	for (uint32_t e = 0; e < n_e_; e++) {
		edges[size_t(e) * 2] = uint32_t(E(e, 0));
		edges[size_t(e) * 2 + 1] = uint32_t(E(e, 1));
	}
	for (uint32_t f = 0; f < n_f_; f++)
		for (int k = 0; k < 3; k++)
			faces[size_t(f) * 3 + size_t(k)] = uint32_t(F(f, k));
	// The queries: garment-involved edges, garment vertices, garment-involved faces.
	queries_.clear();
	for (uint32_t e = 0; e < n_e_; e++)
		if (garment[edges[size_t(e) * 2]] || garment[edges[size_t(e) * 2 + 1]])
			queries_.push_back(e);
	n_q_edges_ = uint32_t(queries_.size());
	for (uint32_t v = 0; v < n_v_; v++)
		if (garment[v])
			queries_.push_back((1u << 30) | v);
	n_q_verts_ = uint32_t(queries_.size()) - n_q_edges_;
	for (uint32_t f = 0; f < n_f_; f++)
		if (garment[faces[size_t(f) * 3]] || garment[faces[size_t(f) * 3 + 1]] || garment[faces[size_t(f) * 3 + 2]])
			queries_.push_back((2u << 30) | f);
	n_q_ = uint32_t(queries_.size());
	if (n_q_ == 0)
		return bail("no garment primitives");

	struct K {
		const char *name;
		::RID *shader, *pipe;
	} ks[2] = { { "swept_aabb", &sh_aabb_, &pipe_aabb_ }, { "box_pair", &sh_pair_, &pipe_pair_ } };
	for (const K &k : ks) {
		size_t bytes = 0;
		const uint8_t *spv = kernel(k.name, bytes);
		if (!spv)
			return bail(std::string(k.name) + " is not embedded");
		*k.shader = dev.shader_from_spirv(spv, bytes);
		if (!k.shader->index)
			return bail(std::string(k.name) + ": shader");
		*k.pipe = dev.compute_pipeline(*k.shader);
		if (!k.pipe->index)
			return bail(std::string(k.name) + ": pipeline");
	}
	x0_.assign(size_t(n_v_) * 3, 0.0f);
	x1_.assign(size_t(n_v_) * 3, 0.0f);
	counts_.assign(n_q_, 0);
	offsets_.assign(n_q_, 0);
	b_x0_ = dev.storage_buffer(x0_.size() * 4, x0_.data());
	b_x1_ = dev.storage_buffer(x1_.size() * 4, x1_.data());
	b_edges_ = dev.storage_buffer(edges.size() * 4, edges.data());
	b_faces_ = dev.storage_buffer(faces.size() * 4, faces.data());
	b_garment_ = dev.storage_buffer(garment.size() * 4, garment.data());
	b_boxes_ = dev.storage_buffer_empty(size_t(n_v_ + n_e_ + n_f_) * 6 * 4);
	b_queries_ = dev.storage_buffer(queries_.size() * 4, queries_.data());
	b_counts_ = dev.storage_buffer(size_t(n_q_) * 4, counts_.data());
	const AabbParams ap{ n_v_, n_e_, n_f_, 0.0f, kMargin, { 0, 0, 0 } };
	const PairParams pc{ n_q_, n_v_, n_e_, n_f_, self_ ? 1u : 0u, 0u, { 0, 0 } };
	const PairParams pw{ n_q_, n_v_, n_e_, n_f_, self_ ? 1u : 0u, 1u, { 0, 0 } };
	u_aabb_ = dev.uniform_buffer(sizeof ap, &ap);
	u_count_ = dev.uniform_buffer(sizeof pc, &pc);
	u_write_ = dev.uniform_buffer(sizeof pw, &pw);
	for (::RID *r : { &b_x0_, &b_x1_, &b_edges_, &b_faces_, &b_garment_, &b_boxes_, &b_queries_, &b_counts_, &u_aabb_, &u_count_, &u_write_ })
		if (!r->index)
			return bail("buffers");
	const int U = rdc::UNIFORM_TYPE_UNIFORM_BUFFER, S = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
	set_aabb_ = dev.uniform_set(sh_aabb_, { { 0, S, b_x0_ }, { 1, S, b_x1_ }, { 2, S, b_edges_ }, { 3, S, b_faces_ }, { 4, S, b_boxes_ }, { 5, U, u_aabb_ } });
	if (!set_aabb_.index)
		return bail("uniform set (aabb)");
	inflation_last_ = 0.0f;
	open_ = true; // grow_pairs needs the flag for close() on failure
	if (!grow_pairs(kPairsInitial))
		return bail("pairs");
	return true;
}

bool GpuBroadPhase::grow_pairs(size_t pairs) {
	rdc::Device &dev = *dev_;
	for (::RID *r : { &set_count_, &set_write_, &b_pairs_ })
		if (r->index) {
			dev.free_rid(*r);
			*r = ::RID();
		}
	pair_cap_ = pairs;
	b_pairs_ = dev.storage_buffer_empty(pair_cap_ * 8);
	if (!b_pairs_.index)
		return fail("pairs buffer");
	const int U = rdc::UNIFORM_TYPE_UNIFORM_BUFFER, S = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
	set_count_ = dev.uniform_set(sh_pair_, { { 0, S, b_queries_ }, { 1, S, b_boxes_ }, { 2, S, b_edges_ }, { 3, S, b_faces_ }, { 4, S, b_garment_ }, { 5, S, b_counts_ }, { 6, S, b_pairs_ }, { 7, U, u_count_ } });
	set_write_ = dev.uniform_set(sh_pair_, { { 0, S, b_queries_ }, { 1, S, b_boxes_ }, { 2, S, b_edges_ }, { 3, S, b_faces_ }, { 4, S, b_garment_ }, { 5, S, b_counts_ }, { 6, S, b_pairs_ }, { 7, U, u_write_ } });
	if (!set_count_.index || !set_write_.index)
		return fail("uniform sets (pairs)");
	stats_.grown++;
	return true;
}

void GpuBroadPhase::close() {
	if (!dev_ || !dev_->ok()) {
		open_ = false;
		return;
	}
	for (::RID *r : { &set_aabb_, &set_count_, &set_write_ })
		if (r->index) {
			dev_->free_rid(*r);
			*r = ::RID();
		}
	for (::RID *r : { &b_x0_, &b_x1_, &b_edges_, &b_faces_, &b_garment_, &b_boxes_, &b_queries_, &b_counts_, &b_pairs_, &u_aabb_, &u_count_, &u_write_,
				 &pipe_aabb_, &pipe_pair_, &sh_aabb_, &sh_pair_ })
		if (r->index) {
			dev_->free_rid(*r);
			*r = ::RID();
		}
	open_ = false;
}

bool GpuBroadPhase::build(const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1, double inflation, ipc::Candidates &out) {
	if (!open_)
		return false;
	rdc::Device &dev = *dev_;
	const int64_t t0 = rdc::host_usec();
	auto give_up = [&](const std::string &what) {
		stats_.fallbacks++;
		return fail(what);
	};
	if (V0.rows() != Eigen::Index(n_v_) || V1.rows() != Eigen::Index(n_v_) || V0.cols() != 3)
		return give_up("vertex count changed");
	for (uint32_t v = 0; v < n_v_; v++)
		for (int d = 0; d < 3; d++) {
			x0_[size_t(v) * 3 + size_t(d)] = float(V0(v, d));
			x1_[size_t(v) * 3 + size_t(d)] = float(V1(v, d));
		}
	if (!dev.buffer_update(b_x0_, 0, x0_.size() * 4, x0_.data()) || !dev.buffer_update(b_x1_, 0, x1_.size() * 4, x1_.data()))
		return give_up("upload");
	if (float(inflation) != inflation_last_) {
		const AabbParams ap{ n_v_, n_e_, n_f_, float(inflation), kMargin, { 0, 0, 0 } };
		if (!dev.buffer_update(u_aabb_, 0, sizeof ap, &ap))
			return give_up("params");
		inflation_last_ = float(inflation);
	}
	// Round trip 1: boxes, then the count per query.
	if (!dev.list_begin())
		return give_up("list_begin");
	dev.bind_pipeline(pipe_aabb_);
	dev.bind_uniform_set(set_aabb_);
	dev.dispatch(rdc::Device::groups_for(n_v_ + n_e_ + n_f_, 64));
	dev.barrier();
	dev.bind_pipeline(pipe_pair_);
	dev.bind_uniform_set(set_count_);
	dev.dispatch(rdc::Device::groups_for(n_q_, 64));
	dev.list_end();
	dev.submit();
	dev.sync();
	if (!dev.buffer_get_into(b_counts_, 0, size_t(n_q_) * 4, counts_.data()))
		return give_up("counts readback");
	const int64_t t1 = rdc::host_usec();
	uint64_t total = 0;
	for (uint32_t q = 0; q < n_q_; q++) {
		offsets_[q] = uint32_t(total);
		total += counts_[q];
	}
	if (total > pair_cap_ && !grow_pairs(size_t(total + total / 2)))
		return give_up("grow");
	const int64_t t2 = rdc::host_usec();
	if (!dev.buffer_update(b_counts_, 0, size_t(n_q_) * 4, offsets_.data()))
		return give_up("offsets upload");
	// Round trip 2: the pairs at their offsets.
	if (!dev.list_begin())
		return give_up("list_begin (write)");
	dev.bind_pipeline(pipe_pair_);
	dev.bind_uniform_set(set_write_);
	dev.dispatch(rdc::Device::groups_for(n_q_, 64));
	dev.list_end();
	dev.submit();
	dev.sync();
	pairs_.resize(size_t(total) * 2);
	if (total > 0 && !dev.buffer_get_into(b_pairs_, 0, size_t(total) * 8, pairs_.data()))
		return give_up("pairs readback");
	const int64_t t3 = rdc::host_usec();
	out.clear();
	for (uint32_t q = 0; q < n_q_; q++) {
		const uint32_t *p = pairs_.data() + size_t(offsets_[q]) * 2;
		if (q < n_q_edges_) {
			for (uint32_t k = 0; k < counts_[q]; k++)
				out.ee_candidates.emplace_back(long(p[2 * k]), long(p[2 * k + 1]));
		} else {
			for (uint32_t k = 0; k < counts_[q]; k++)
				out.fv_candidates.emplace_back(long(p[2 * k]), long(p[2 * k + 1]));
		}
	}
	const int64_t t4 = rdc::host_usec();
	stats_.builds++;
	stats_.gpu_us += (t1 - t0) + (t3 - t2);
	stats_.convert_us += (t2 - t1) + (t4 - t3);
	stats_.last_pairs = int64_t(total);
	stats_.pairs += int64_t(total);
	stats_.last_us = t4 - t0;
	stats_.total_us += stats_.last_us;

	if (audit_) {
		ipc::Candidates ref;
		const int64_t a0 = rdc::host_usec();
		ref.build(*mesh_, V0, V1, inflation, method_);
		stats_.audit_cpu_us += rdc::host_usec() - a0;
		stats_.audit_builds++;
		stats_.audit_cpu_pairs += int64_t(ref.ee_candidates.size() + ref.fv_candidates.size());
		std::unordered_set<uint64_t> gpu_ee, gpu_fv;
		gpu_ee.reserve(out.ee_candidates.size() * 2);
		gpu_fv.reserve(out.fv_candidates.size() * 2);
		for (const auto &c : out.ee_candidates)
			gpu_ee.insert(key(uint32_t(std::min(c.edge0_id, c.edge1_id)), uint32_t(std::max(c.edge0_id, c.edge1_id))));
		for (const auto &c : out.fv_candidates)
			gpu_fv.insert(key(uint32_t(c.face_id), uint32_t(c.vertex_id)));
		std::unordered_set<uint64_t> cpu_ee, cpu_fv;
		cpu_ee.reserve(ref.ee_candidates.size() * 2);
		cpu_fv.reserve(ref.fv_candidates.size() * 2);
		for (const auto &c : ref.ee_candidates)
			cpu_ee.insert(key(uint32_t(std::min(c.edge0_id, c.edge1_id)), uint32_t(std::max(c.edge0_id, c.edge1_id))));
		for (const auto &c : ref.fv_candidates)
			cpu_fv.insert(key(uint32_t(c.face_id), uint32_t(c.vertex_id)));
		for (uint64_t k : cpu_ee)
			if (!gpu_ee.count(k))
				stats_.audit_missed++;
		for (uint64_t k : cpu_fv)
			if (!gpu_fv.count(k))
				stats_.audit_missed++;
		for (uint64_t k : gpu_ee)
			if (!cpu_ee.count(k))
				stats_.audit_extra++;
		for (uint64_t k : gpu_fv)
			if (!cpu_fv.count(k))
				stats_.audit_extra++;
	}
	return true;
}

} // namespace fit_gpu
