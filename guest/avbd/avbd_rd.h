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
//    list, which is where the per-submit cost is amortised. The backward
//    pass and the self-collision scan submit the same way (AGENTS.md rule 4):
//    the caller reads back on a later tick, when pending() is known done;
//  - ragged tails are padded as upstream's AvbdSolverVk does: every vertex
//    and constraint buffer holds cap_for(n) = roundUp(n + 1, 64) elements,
//    index buffers are padded with the dummy vertex nV and stiffnesses with
//    0, and hScratch's pad rows are the identity, so the kernels without a
//    `lane >= count` guard (the force kernels and most of the backward) read
//    and write only dummy slots past n. vertPerm is not padded: every kernel
//    that reads it is guarded.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
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
	// The rest-shape update (Cut 6d's similarity fit): new rest metrics,
	// stiffnesses and radii on the SAME topology, written into the existing
	// buffers (buffer_update). No RID is created and no uniform set is rebuilt:
	// a re-upload costs ~300 permanent RID slots (buffers + kernels x colours
	// sets), and the sandbox's permanent table filled after ~600 of them
	// (Gate 6d pass 3). The duals keep their values; gamma follows k.
	void updateTriangleRest(const float *invUV, const float *stiffness);
	void updateBendingRest(const float *weight, const float *nTarget, const float *stiffness);
	void updateAttachmentStiffness(const float *stiffness);
	void updateSelfCollisionRadii(const float *radii);

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
	// A submit is in flight: sync (or any readback) waits for it.
	bool pending() const { return pending_; }

	void readPositions(std::vector<float> &out);

	// Reverse-mode adjoint of the last recorded iteration
	// (avbd_rd_backward.cpp); accessors laid out as cloth::AvbdSolver's.
	int stepBackward(const float *v_positions_loss);
	// run(iters, duals) and stepBackward(v) recorded into one graph and ONE
	// submit; the backward differentiates the last iteration.
	int runWithBackward(int iters, bool duals, const float *v_positions_loss);
	void readPositionsGrad(std::vector<float> &out);
	void readMassGrad(std::vector<float> &out);
	void readPredictedGrad(std::vector<float> &out);
	void readSpringGrad(std::vector<float> &restLen_grad, std::vector<float> &stiff_grad);
	void readAttachGrad(std::vector<float> &fixedPos_grad, std::vector<float> &stiff_grad,
			std::vector<float> &lambda_grad);
	void readTriGrad(std::vector<float> &stiff_grad, std::vector<float> &lambda0_grad,
			std::vector<float> &lambda1_grad);
	void readBendGrad(std::vector<float> &nTarget_grad, std::vector<float> &stiff_grad,
			std::vector<float> &lambda_grad);

	// Self-collision scan (avbd_rd_backward.cpp).
	// submitSelfCollisionScan submits and returns; collectSelfCollisions
	// reads the pairs (syncing if still pending: call it on a later tick).
	// There is deliberately no one-call detectSelfCollisions here (AvbdCpu
	// has one): on the GPU it would sync in its submit's frame (rule 4).
	void uploadSelfCollisionRadii(const float *radii, uint32_t maxNeighborsPerVert);
	int submitSelfCollisionScan();
	int collectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs);

	// The index every index buffer's padding points at; by default the dummy
	// vertex nV. 0 points the ragged-tail lanes at a real vertex, which
	// reproduces the race the padding exists to prevent (the negative
	// control). Applies to uploads made after the call.
	void setPadFillForTest(uint32_t fill) {
		padOverride_ = true;
		padFill_ = fill;
	}

	// Test hook for gradcheck_duals' negative control: false makes the
	// backward bind the live duals instead of the pre-step copies, which is
	// wrong once a dual update has run after the step.
	void setLambdaSnapshotForTest(bool use) {
		useLambdaSnapshot_ = use;
		invalidate_sets();
	}

	// Test hooks for the per-kernel bisection (drape job mesh_bisect): with a
	// stop set, the LAST iteration of run() records colours before `color`
	// whole and colour `color` up to `stage` (0 init and the four force
	// kernels, 1-4 the spring/attachment/triangle/bending gathers, 5 the
	// solve), and no dual update after it; -1 records whole iterations.
	// readDebugForTest reads a buffer by the kernels' name (positions,
	// gScratch, hScratch, attachGradV, attachHess, triGrad, triHess, bendGrad,
	// bendHess), float3 rows compacted to xyz.
	void setDebugStopForTest(int color, int stage) {
		dbgColor_ = color;
		dbgStage_ = stage;
	}
	std::vector<float> readDebugForTest(const std::string &name);

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
	// Colour indices that select a params block in set_for other than a
	// colour's own: the whole-mesh scatter, and the three saxpby uses.
	static constexpr int kScatterColor = -2;
	static constexpr int kSaxDelta = -10, kSaxSum = -11, kSaxOut = -12;

	bool load_kernels();
	Buf make_f32(const std::vector<float> &v);
	Buf make_u32(const std::vector<uint32_t> &v);
	Buf make_v3(const float *xyz, uint32_t n); // padded to 16 bytes
	Buf make_zero(size_t bytes);
	// Padded to `cap` elements: zeros, or `fill` for index buffers.
	Buf make_f32_cap(const float *v, uint32_t n, uint32_t cap);
	Buf make_u32_cap(const uint32_t *v, uint32_t n, uint32_t cap, uint32_t fill);
	Buf make_v3_cap(const float *xyz, uint32_t n, uint32_t cap);
	static uint32_t cap_for(uint32_t n) { return ((n + 1 + 63) / 64) * 64; }
	uint32_t pad_index() const { return padOverride_ ? padFill_ : nVerts_; }
	void update_v3(Buf &b, const float *xyz, uint32_t n);
	void update_f32(Buf &b, const std::vector<float> &v);
	void free_buf(Buf &b);
	void invalidate_sets();
	void ensure_params();
	::RID set_for(const char *kernel, int color, const Binds &binds);
	bool dispatch(const char *kernel, int color, uint32_t threads, const Binds &binds);
	// `last`: the debug stop applies (setDebugStopForTest).
	void record_iteration(bool last = false);
	void record_duals();
	bool begin_record();
	void end_record_and_submit();
	// Checks, params, waits out a previous tick's submit. False if not set up.
	bool prologue();
	// Iterations 1..n-1 in one list, the pre-step snapshot, then the last
	// iteration in a list left open for the caller to end or extend.
	void record_run(int iters, bool duals);
	// The backward's transfers (cotangent upload, clears): before any list.
	bool prepare_backward(const float *v_positions_loss);
	// The backward's dispatches, into the open list.
	void record_backward();
	// Forward gather kernels used as scatters in the backward pass: a
	// whole-mesh params block and cotangent buffers in place of gradients.
	bool dispatch_scatter(const char *kernel, uint32_t threads, const Binds &binds);
	void snapshot_pre_step();
	void mark_lambdas_dirty() { lambdasDirty_ = true; }
	bool ensure_backward_buffers();
	std::vector<float> read_v3(const Buf &b, uint32_t n);
	std::vector<float> read_f32(const Buf &b, uint32_t n);

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
	bool padOverride_ = false;
	uint32_t padFill_ = 0;
	int dbgColor_ = -1, dbgStage_ = -1;

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

	// Backward: the pre-step positions and duals (copied before the last
	// recorded iteration; the duals only when a dual update ran since the
	// last copy) and every cotangent, plus the whole-mesh scatter params, the
	// init-backward params and the three saxpby params (deltaX = x - xPre;
	// sum = vPosInit + vOut; out = vPosGrad + sum).
	bool backwardReady_ = false;
	bool hasStep_ = false;
	bool lambdasDirty_ = false;
	bool useLambdaSnapshot_ = true;
	Buf positionsPre_, tri_l0Pre_, tri_l1Pre_, bd_lambdaPre_;
	Buf vOut_, vG_, vH_, deltaX_, vPosGrad_, vPosInit_, vPred_, vMass_, hJunk_, vPosSum_, vPosGradOut_;
	Buf vSpringGradA_, vSpringHess_, vSpringPd_, vSpringRest_, vSpringStiff_;
	Buf vAttachGradV_, vAttachHess_, vAttachFixed_, vAttachStiff_, vAttachLambda_;
	Buf vTriGrad_, vTriHess_, vTriP_, vTriStiff_, vTriL0_, vTriL1_;
	Buf vBendGrad_, vBendHess_, vBendP_, vBendN_, vBendStiff_, vBendLambda_;
	Buf scatterParams_, initBwdParams_, sxDeltaParams_, sxSumParams_, sxOutParams_;
	// Self-collision.
	Buf radii_, neighbors_, selfParams_;
	uint32_t selfK_ = 0;
};
