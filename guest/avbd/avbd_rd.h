// AvbdRd -- the AVBD kernels on the GPU through rd_compute.
//
// The same Lean-emitted kernels AvbdCpu runs one thread at a time, as SPIR-V
// dispatched through Godot's RenderingDevice, with the same public surface as
// AvbdCpu so the driver (avbd_sim.h) and the fixture are templated over the
// backend. The dispatch sequence mirrors avbd_cpu.cpp line for line; buffer
// names and binding slots come from the generated table (avbd_table.h), so
// the port cannot drift from the kernels.
//
// What is different from the CPU path, all forced by the layer beneath:
//  - float3 buffers are 16-byte strided (std430), so xyz triples are padded
//    on upload and compacted on readback;
//  - params blocks live in per-(kernel, colour) uniform buffers, filled when
//    the colouring is built, because buffer_update is refused inside a
//    compute list;
//  - one uniform set per (kernel, colour), cached, rebuilt after any buffer
//    is reallocated;
//  - a barrier after every dispatch except between the four force kernels of
//    one colour, whose outputs are disjoint;
//  - step() records and SUBMITS but does not wait. sync() waits; readbacks
//    sync if a submit is pending. run(iters) records many iterations in one
//    list, which is where the per-submit cost is amortised.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../rd_compute.h"
#include "../avbd_table.h"

class AvbdRd {
public:
	explicit AvbdRd(rdc::Device &dev);
	~AvbdRd();

	// True once every forward/dual kernel has a shader and pipeline.
	bool ok() const { return ok_; }
	const std::string &error() const { return err_; }

	void setupMesh(uint32_t nVerts, const float *positions, const float *predicted,
			const float *mass, float invHSquared);
	void uploadSprings(uint32_t nSprings, const uint32_t *p1Idx, const uint32_t *p2Idx,
			const float *restLen, const float *stiffness);
	void uploadAttachments(uint32_t nAttach, const uint32_t *vertIdx,
			const float *fixedPos, const float *stiffness);
	void uploadTriangles(uint32_t nTri, const uint32_t *triIdx, const float *invUV,
			const float *stiffness);
	void uploadBendings(uint32_t nBend, const uint32_t *bendIdx, const float *weight,
			const float *nTarget, const float *stiffness);

	void buildColoring();
	void setGammaScale(float scale);
	void setPenaltyRamp(float beta, float penaltyMax);
	void updateState(const float *positions, const float *predicted);
	void updateAttachmentFixedPos(const float *fixedPos);

	// One outer iteration: record every colour, submit, return. -1 if not set up.
	int step();
	int stepDualAttachments();
	int stepDualMembrane();
	int stepDualBending();
	// `iters` outer iterations, each followed by the three dual updates when
	// `duals` is set, in ONE compute list and one submit.
	int run(int iters, bool duals);
	// Wait for the pending submit, if any.
	void sync();

	void readPositions(std::vector<float> &out);
	uint32_t nVerts() const { return nVerts_; }
	bool ready() const { return meshReady_; }
	uint32_t numColors() const { return uint32_t(colorOffsets_.size()) - 1; }

private:
	struct Buf {
		::RID rid;
		size_t bytes = 0;
		bool valid() const { return rid.index != 0; }
	};
	struct Kernel {
		::RID shader;
		::RID pipeline;
		const cloth::avbd_table::KernelDesc *desc = nullptr;
	};
	// A binding-name -> buffer list for one kernel; the mirror of avbd_cpu's
	// `gp.X_0.data = ...` lines.
	using Binds = std::vector<std::pair<const char *, const Buf *>>;

	bool load_kernels();
	Buf make_f32(const std::vector<float> &v);
	Buf make_u32(const std::vector<uint32_t> &v);
	Buf make_v3(const float *xyz, uint32_t n); // padded to 16 bytes
	Buf make_zero(size_t bytes);
	void update_v3(Buf &b, const float *xyz, uint32_t n);
	void update_f32(Buf &b, const std::vector<float> &v);
	void free_buf(Buf &b);
	void invalidate_sets();
	void ensure_params();
	::RID set_for(const char *kernel, int color, const Binds &binds);
	bool dispatch(const char *kernel, int color, uint32_t threads, const Binds &binds);
	void record_iteration();
	void record_duals();
	bool begin_record();
	void end_record_and_submit();

	rdc::Device &d_;
	bool ok_ = false;
	std::string err_;
	std::map<std::string, Kernel> kernels_;
	std::map<std::string, ::RID> sets_;

	uint32_t nVerts_ = 0, nSprings_ = 0, nAttach_ = 0, nTri_ = 0, nBend_ = 0;
	float invHSq_ = 0.0f;
	float beta_ = 0.0f, penaltyMax_ = 1e10f;
	float gammaScale_ = 1.0f;
	bool meshReady_ = false;
	bool pending_ = false;
	bool paramsReady_ = false;

	// CPU-side topology and the arrays that seed the GPU buffers.
	std::vector<uint32_t> vertPerm_, colorOffsets_{ 0u, 0u };
	std::vector<uint32_t> springP1_, springP2_, triIdx_, bendIdx_;
	std::vector<float> attachGamma_, triGamma_, bendGamma_;

	// GPU buffers, named as the kernels name them.
	Buf positions_, predicted_, mass_, gScratch_, hScratch_, vertPerm_b_;
	Buf sp_p1_, sp_p2_, sp_rest_, sp_k_, sp_gradA_, sp_hess_, sp_off_, sp_idx_, sp_role_;
	Buf at_vert_, at_fixed_, at_k_, at_lambda_, at_gamma_, at_gradV_, at_hess_, at_off_, at_idx_;
	Buf tri_idx_, tri_k_, tri_l0_, tri_l1_, tri_gamma_, tri_grad_, tri_hess_, tri_invuv_, tri_off_, tri_idxc_, tri_role_;
	Buf bd_idx_, bd_w_, bd_n_, bd_k_, bd_lambda_, bd_gamma_, bd_grad_, bd_hess_, bd_off_, bd_idxc_, bd_role_;
	// Params: per colour {invHSq, offset, count} for init and {offset, count}
	// for the gathers and solve_apply (same layout, shared); one per family
	// for the dual updates {beta, penaltyMax, count}.
	std::vector<Buf> initParams_, colorParams_;
	Buf dualAttachParams_, dualTriParams_, dualBendParams_;
};
