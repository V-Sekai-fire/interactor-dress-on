// AvbdCpu: the reverse-mode adjoint and the self-collision scan, on the same
// Lean-emitted kernels compiled to C++. The sequence mirrors cloth-dynamics'
// AvbdSolverVk::stepBackward: solve_apply backward -> per-family gather
// backwards -> per-family force backwards (on the PRE-step positions) ->
// init backward -> scatter the per-constraint position cotangents into
// v_positions by reusing the forward gather kernels -> epilogue adding the
// inertial and direct paths. The prologue delta and the epilogue sums are the
// emitted saxpby, as on AvbdRd, so both backends run the same kernel.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "avbd_cpu.h"

#include <algorithm>
#include <cstring>

#include "slang-cpp-prelude.h"

#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace kb_sab {
#include "vbd_solve_apply_backward_emit.cpp"
}
namespace kb_gsb {
#include "vbd_gather_spring_backward_emit.cpp"
}
namespace kb_gab {
#include "vbd_gather_attachment_backward_emit.cpp"
}
namespace kb_gtb {
#include "vbd_gather_triangle_backward_emit.cpp"
}
namespace kb_gbb {
#include "vbd_gather_bending_backward_emit.cpp"
}
namespace kb_afab {
#include "attachment_force_al_backward_emit.cpp"
}
namespace kb_sfb {
#include "spring_force_backward_emit.cpp"
}
namespace kb_tmfb {
#include "triangle_membrane_force_al_backward_emit.cpp"
}
namespace kb_tbfb {
#include "triangle_bending_force_al_backward_emit.cpp"
}
namespace kb_ib {
#include "vbd_init_backward_emit.cpp"
}
// The forward gathers again, for the scatter (a second copy in this TU,
// namespaced apart from avbd_cpu.cpp's).
namespace kb_gs {
#include "vbd_gather_spring_emit.cpp"
}
namespace kb_gt {
#include "vbd_gather_triangle_emit.cpp"
}
namespace kb_gb {
#include "vbd_gather_bending_emit.cpp"
}
namespace kb_scs {
#include "self_collision_scan_emit.cpp"
}
namespace kb_sax {
#include "saxpby_emit.cpp"
}

namespace {

using Vec3 = Vector<float, 3>;
using ThreadFn = void (*)(ComputeThreadVaryingInput *, void *, void *);

inline void dispatch(uint32_t count, ThreadFn fn, void *gp) {
	for (uint32_t lane = 0; lane < count; ++lane) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(lane, 0u, 0u);
		fn(&t, nullptr, gp);
	}
}

// dst = alpha * x + beta * y over n floats (dst may alias x or y: each lane
// reads its x[i], y[i] before writing dst[i]).
void saxpby(std::vector<float> &dst, std::vector<float> &x, std::vector<float> &y, uint32_t n, float alpha,
		float beta) {
	kb_sax::GlobalParams_0 gp{};
	kb_sax::SaxpbyParams_0 p{ n, alpha, beta };
	gp.params_0 = &p;
	gp.x_0.data = x.data(); gp.x_0.count = n;
	gp.y_0.data = y.data(); gp.y_0.count = n;
	gp.dst_0.data = dst.data(); gp.dst_0.count = n;
	dispatch(n, &kb_sax::main_0_Thread, &gp);
}

inline Vec3 *v3(std::vector<float> &b) { return reinterpret_cast<Vec3 *>(b.data()); }
inline const Vec3 *v3(const std::vector<float> &b) { return reinterpret_cast<const Vec3 *>(b.data()); }

constexpr uint32_t kSentinel = 0xFFFFFFFFu;

} // namespace

int AvbdCpu::stepBackward(const float *v_positions_loss) {
	if (!meshReady_ || positionsPre_.size() != positions_.size()) {
		return -1;
	}
	const uint32_t nV = nVerts_;

	// Prologue: the incoming cotangent, the forward step's position delta,
	// and the accumulators the scatter adds into.
	vOut_.assign(v_positions_loss, v_positions_loss + 3 * nV);
	deltaX_.resize(3 * nV);
	saxpby(deltaX_, positions_, positionsPre_, 3 * nV, 1.0f, -1.0f);
	vG_.assign(3 * nV, 0.0f);
	vH_.assign(6 * nV, 0.0f);
	vPosGrad_.assign(3 * nV, 0.0f);
	vPosInit_.assign(3 * nV, 0.0f);
	vPred_.assign(3 * nV, 0.0f);
	vMass_.assign(nV, 0.0f);
	hJunk_.assign(6 * nV, 0.0f);
	vSpringGradA_.assign(3 * nSprings_, 0.0f);
	vSpringHess_.assign(6 * nSprings_, 0.0f);
	vSpringPd_.assign(3 * nSprings_, 0.0f);
	vSpringRest_.assign(nSprings_, 0.0f);
	vSpringStiff_.assign(nSprings_, 0.0f);
	vAttachGradV_.assign(3 * nAttach_, 0.0f);
	vAttachHess_.assign(nAttach_, 0.0f);
	vAttachFixed_.assign(3 * nAttach_, 0.0f);
	vAttachStiff_.assign(nAttach_, 0.0f);
	vAttachLambda_.assign(3 * nAttach_, 0.0f);
	vTriGrad_.assign(3 * 3 * nTri_, 0.0f);
	vTriHess_.assign(3 * nTri_, 0.0f);
	vTriP_.assign(3 * 3 * nTri_, 0.0f);
	vTriStiff_.assign(nTri_, 0.0f);
	vTriL0_.assign(3 * nTri_, 0.0f);
	vTriL1_.assign(3 * nTri_, 0.0f);
	vBendGrad_.assign(3 * 4 * nBend_, 0.0f);
	vBendHess_.assign(4 * nBend_, 0.0f);
	vBendP_.assign(3 * 4 * nBend_, 0.0f);
	vBendN_.assign(nBend_, 0.0f);
	vBendStiff_.assign(nBend_, 0.0f);
	vBendLambda_.assign(3 * nBend_, 0.0f);

	{
		kb_sab::GlobalParams_0 gp{};
		gp.v_out_0.data = v3(vOut_); gp.v_out_0.count = nV;
		gp.hScratch_0.data = hScratch_.data(); gp.hScratch_0.count = 6 * nV;
		gp.deltaX_0.data = v3(deltaX_); gp.deltaX_0.count = nV;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		dispatch(nV, &kb_sab::main_0_Thread, &gp);
	}
	if (nSprings_) {
		kb_gsb::GlobalParams_0 gp{};
		gp.springP1Idx_0.data = springP1_.data(); gp.springP1Idx_0.count = nSprings_;
		gp.springP2Idx_0.data = springP2_.data(); gp.springP2Idx_0.count = nSprings_;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		gp.v_springGradA_0.data = v3(vSpringGradA_); gp.v_springGradA_0.count = nSprings_;
		gp.v_springHess_0.data = vSpringHess_.data(); gp.v_springHess_0.count = 6 * nSprings_;
		dispatch(nSprings_, &kb_gsb::main_0_Thread, &gp);
	}
	if (nAttach_) {
		kb_gab::GlobalParams_0 gp{};
		gp.attachVertIdx_0.data = attachVert_.data(); gp.attachVertIdx_0.count = nAttach_;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		gp.v_attachGradV_0.data = v3(vAttachGradV_); gp.v_attachGradV_0.count = nAttach_;
		gp.v_attachHessScalar_0.data = vAttachHess_.data(); gp.v_attachHessScalar_0.count = nAttach_;
		dispatch(nAttach_, &kb_gab::main_0_Thread, &gp);
	}
	if (nTri_) {
		kb_gtb::GlobalParams_0 gp{};
		gp.triIdx_0.data = triIdx_.data(); gp.triIdx_0.count = 3 * nTri_;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		gp.v_triGrad_0.data = v3(vTriGrad_); gp.v_triGrad_0.count = 3 * nTri_;
		gp.v_triHessScalar_0.data = vTriHess_.data(); gp.v_triHessScalar_0.count = 3 * nTri_;
		dispatch(3 * nTri_, &kb_gtb::main_0_Thread, &gp);
	}
	if (nBend_) {
		kb_gbb::GlobalParams_0 gp{};
		gp.bendIdx_0.data = bendIdx_.data(); gp.bendIdx_0.count = 4 * nBend_;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		gp.v_bendGrad_0.data = v3(vBendGrad_); gp.v_bendGrad_0.count = 4 * nBend_;
		gp.v_bendHessScalar_0.data = vBendHess_.data(); gp.v_bendHessScalar_0.count = 4 * nBend_;
		dispatch(4 * nBend_, &kb_gbb::main_0_Thread, &gp);
	}

	// Force backwards see the pre-step positions. attachment's v_positions
	// write is 1:1 (non-additive) and lands before the additive scatters.
	if (nAttach_) {
		kb_afab::GlobalParams_0 gp{};
		gp.positions_0.data = v3(positionsPre_); gp.positions_0.count = nV;
		gp.vertIdx_0.data = attachVert_.data(); gp.vertIdx_0.count = nAttach_;
		gp.fixedPos_0.data = v3(attachFixed_); gp.fixedPos_0.count = nAttach_;
		gp.stiffness_0.data = attachStiff_.data(); gp.stiffness_0.count = nAttach_;
		gp.v_gradV_0.data = v3(vAttachGradV_); gp.v_gradV_0.count = nAttach_;
		gp.v_hessScalar_0.data = vAttachHess_.data(); gp.v_hessScalar_0.count = nAttach_;
		gp.v_positions_0.data = v3(vPosGrad_); gp.v_positions_0.count = nV;
		gp.v_fixedPos_0.data = v3(vAttachFixed_); gp.v_fixedPos_0.count = nAttach_;
		gp.v_lambda_0.data = v3(vAttachLambda_); gp.v_lambda_0.count = nAttach_;
		gp.v_stiffness_0.data = vAttachStiff_.data(); gp.v_stiffness_0.count = nAttach_;
		dispatch(nAttach_, &kb_afab::main_0_Thread, &gp);
	}
	if (nSprings_) {
		kb_sfb::GlobalParams_0 gp{};
		gp.positions_0.data = v3(positionsPre_); gp.positions_0.count = nV;
		gp.p1Idx_0.data = springP1_.data(); gp.p1Idx_0.count = nSprings_;
		gp.p2Idx_0.data = springP2_.data(); gp.p2Idx_0.count = nSprings_;
		gp.restLen_0.data = springRest_.data(); gp.restLen_0.count = nSprings_;
		gp.stiffness_0.data = springStiff_.data(); gp.stiffness_0.count = nSprings_;
		gp.v_gradA_0.data = v3(vSpringGradA_); gp.v_gradA_0.count = nSprings_;
		gp.v_springHess_0.data = vSpringHess_.data(); gp.v_springHess_0.count = 6 * nSprings_;
		gp.v_p_d_0.data = v3(vSpringPd_); gp.v_p_d_0.count = nSprings_;
		gp.v_restLen_0.data = vSpringRest_.data(); gp.v_restLen_0.count = nSprings_;
		gp.v_stiffness_0.data = vSpringStiff_.data(); gp.v_stiffness_0.count = nSprings_;
		dispatch(nSprings_, &kb_sfb::main_0_Thread, &gp);
	}
	if (nTri_) {
		kb_tmfb::GlobalParams_0 gp{};
		gp.positions_0.data = v3(positionsPre_); gp.positions_0.count = nV;
		gp.idx_0.data = triIdx_.data(); gp.idx_0.count = 3 * nTri_;
		gp.stiffness_0.data = triStiff_.data(); gp.stiffness_0.count = nTri_;
		gp.lambda0_0.data = v3(useLambdaSnapshot_ ? triLambda0Pre_ : triLambda0_); gp.lambda0_0.count = nTri_;
		gp.lambda1_0.data = v3(useLambdaSnapshot_ ? triLambda1Pre_ : triLambda1_); gp.lambda1_0.count = nTri_;
		gp.inv_deltaUV_0.data = triInvUV_.data(); gp.inv_deltaUV_0.count = 4 * nTri_;
		gp.v_grad_0.data = v3(vTriGrad_); gp.v_grad_0.count = 3 * nTri_;
		gp.v_hessScalar_0.data = vTriHess_.data(); gp.v_hessScalar_0.count = 3 * nTri_;
		gp.v_p_0.data = v3(vTriP_); gp.v_p_0.count = 3 * nTri_;
		gp.v_stiffness_0.data = vTriStiff_.data(); gp.v_stiffness_0.count = nTri_;
		gp.v_lambda0_0.data = v3(vTriL0_); gp.v_lambda0_0.count = nTri_;
		gp.v_lambda1_0.data = v3(vTriL1_); gp.v_lambda1_0.count = nTri_;
		dispatch(nTri_, &kb_tmfb::main_0_Thread, &gp);
	}
	if (nBend_) {
		kb_tbfb::GlobalParams_0 gp{};
		gp.positions_0.data = v3(positionsPre_); gp.positions_0.count = nV;
		gp.idx_0.data = bendIdx_.data(); gp.idx_0.count = 4 * nBend_;
		gp.weight_0.data = bendWeight_.data(); gp.weight_0.count = 4 * nBend_;
		gp.nTarget_0.data = bendNTarget_.data(); gp.nTarget_0.count = nBend_;
		gp.stiffness_0.data = bendStiff_.data(); gp.stiffness_0.count = nBend_;
		gp.lambda_0.data = v3(useLambdaSnapshot_ ? bendLambdaPre_ : bendLambda_); gp.lambda_0.count = nBend_;
		gp.v_grad_0.data = v3(vBendGrad_); gp.v_grad_0.count = 4 * nBend_;
		gp.v_hessScalar_0.data = vBendHess_.data(); gp.v_hessScalar_0.count = 4 * nBend_;
		gp.v_p_0.data = v3(vBendP_); gp.v_p_0.count = 4 * nBend_;
		gp.v_nTarget_0.data = vBendN_.data(); gp.v_nTarget_0.count = nBend_;
		gp.v_stiffness_0.data = vBendStiff_.data(); gp.v_stiffness_0.count = nBend_;
		gp.v_lambda_0.data = v3(vBendLambda_); gp.v_lambda_0.count = nBend_;
		dispatch(nBend_, &kb_tbfb::main_0_Thread, &gp);
	}
	{
		kb_ib::GlobalParams_0 gp{};
		kb_ib::VbdInitBackwardParams_0 ip{ invHSq_ };
		gp.params_0 = &ip;
		gp.positions_0.data = v3(positionsPre_); gp.positions_0.count = nV;
		gp.predicted_0.data = v3(predicted_); gp.predicted_0.count = nV;
		gp.mass_0.data = mass_.data(); gp.mass_0.count = nV;
		gp.v_g_0.data = v3(vG_); gp.v_g_0.count = nV;
		gp.v_H_0.data = vH_.data(); gp.v_H_0.count = 6 * nV;
		gp.v_x_0.data = v3(vPosInit_); gp.v_x_0.count = nV;
		gp.v_y_0.data = v3(vPred_); gp.v_y_0.count = nV;
		gp.v_mass_0.data = vMass_.data(); gp.v_mass_0.count = nV;
		dispatch(nV, &kb_ib::main_0_Thread, &gp);
	}

	// Scatter: the forward gathers over the whole mesh (colorOffset 0,
	// count nV), fed the per-constraint position cotangents in place of the
	// per-constraint gradients, accumulating into v_positions. The
	// permutation is the identity over every vertex (identPerm_), not the
	// forward's vertPerm_, which restrictToOwned shortens.
	if (nSprings_) {
		kb_gs::GlobalParams_0 gg{};
		kb_gs::VbdGatherSpringParams_0 gpar{ 0u, nV };
		gg.springGradA_0.data = v3(vSpringPd_); gg.springGradA_0.count = nSprings_;
		gg.springHess_0.data = vSpringHess_.data(); gg.springHess_0.count = 6 * nSprings_;
		gg.vertSpringOffset_0.data = vSpringOff_.data(); gg.vertSpringOffset_0.count = nV + 1;
		gg.vertSpringIdx_0.data = vSpringIdx_.data(); gg.vertSpringIdx_0.count = (uint32_t)vSpringIdx_.size();
		gg.vertSpringRole_0.data = vSpringRole_.data(); gg.vertSpringRole_0.count = (uint32_t)vSpringRole_.size();
		gg.gScratch_0.data = v3(vPosGrad_); gg.gScratch_0.count = nV;
		gg.hScratch_0.data = hJunk_.data(); gg.hScratch_0.count = 6 * nV;
		gg.vertPerm_0.data = identPerm_.data(); gg.vertPerm_0.count = nV;
		gg.params_0 = &gpar;
		dispatch(nV, &kb_gs::main_0_Thread, &gg);
	}
	if (nTri_) {
		kb_gt::GlobalParams_0 gg{};
		kb_gt::VbdGatherTriangleParams_0 gpar{ 0u, nV };
		gg.triGrad_0.data = v3(vTriP_); gg.triGrad_0.count = 3 * nTri_;
		gg.triHessScalar_0.data = vTriHess_.data(); gg.triHessScalar_0.count = 3 * nTri_;
		gg.vertTriOffset_0.data = vTriOff_.data(); gg.vertTriOffset_0.count = nV + 1;
		gg.vertTriIdx_0.data = vTriIdx_.data(); gg.vertTriIdx_0.count = (uint32_t)vTriIdx_.size();
		gg.vertTriRole_0.data = vTriRole_.data(); gg.vertTriRole_0.count = (uint32_t)vTriRole_.size();
		gg.gScratch_0.data = v3(vPosGrad_); gg.gScratch_0.count = nV;
		gg.hScratch_0.data = hJunk_.data(); gg.hScratch_0.count = 6 * nV;
		gg.vertPerm_0.data = identPerm_.data(); gg.vertPerm_0.count = nV;
		gg.params_0 = &gpar;
		dispatch(nV, &kb_gt::main_0_Thread, &gg);
	}
	if (nBend_) {
		kb_gb::GlobalParams_0 gg{};
		kb_gb::VbdGatherBendingParams_0 gpar{ 0u, nV };
		gg.bendGrad_0.data = v3(vBendP_); gg.bendGrad_0.count = 4 * nBend_;
		gg.bendHessScalar_0.data = vBendHess_.data(); gg.bendHessScalar_0.count = 4 * nBend_;
		gg.vertBendOffset_0.data = vBendOff_.data(); gg.vertBendOffset_0.count = nV + 1;
		gg.vertBendIdx_0.data = vBendIdx_.data(); gg.vertBendIdx_0.count = (uint32_t)vBendIdx_.size();
		gg.vertBendRole_0.data = vBendRole_.data(); gg.vertBendRole_0.count = (uint32_t)vBendRole_.size();
		gg.gScratch_0.data = v3(vPosGrad_); gg.gScratch_0.count = nV;
		gg.hScratch_0.data = hJunk_.data(); gg.hScratch_0.count = 6 * nV;
		gg.vertPerm_0.data = identPerm_.data(); gg.vertPerm_0.count = nV;
		gg.params_0 = &gpar;
		dispatch(nV, &kb_gb::main_0_Thread, &gg);
	}

	// Epilogue: the inertial path (v_x from init backward) and the direct
	// path (x_out = p + dx carries the incoming cotangent unchanged), summed
	// in AvbdRd's order: vPosSum = vPosInit + vOut, then vPosGrad += vPosSum.
	vPosSum_.resize(3 * nV);
	saxpby(vPosSum_, vPosInit_, vOut_, 3 * nV, 1.0f, 1.0f);
	saxpby(vPosGrad_, vPosGrad_, vPosSum_, 3 * nV, 1.0f, 1.0f);
	backwardReady_ = true;
	return 0;
}

void AvbdCpu::readPositionsGrad(std::vector<float> &out) const { out = vPosGrad_; }
void AvbdCpu::readMassGrad(std::vector<float> &out) const { out = vMass_; }
void AvbdCpu::readPredictedGrad(std::vector<float> &out) const { out = vPred_; }
void AvbdCpu::readSpringGrad(std::vector<float> &restLen_grad, std::vector<float> &stiff_grad) const {
	restLen_grad = vSpringRest_;
	stiff_grad = vSpringStiff_;
}
void AvbdCpu::readAttachGrad(std::vector<float> &fixedPos_grad, std::vector<float> &stiff_grad,
		std::vector<float> &lambda_grad) const {
	fixedPos_grad = vAttachFixed_;
	stiff_grad = vAttachStiff_;
	lambda_grad = vAttachLambda_;
}
void AvbdCpu::readTriGrad(std::vector<float> &stiff_grad, std::vector<float> &lambda0_grad,
		std::vector<float> &lambda1_grad) const {
	stiff_grad = vTriStiff_;
	lambda0_grad = vTriL0_;
	lambda1_grad = vTriL1_;
}
void AvbdCpu::readBendGrad(std::vector<float> &nTarget_grad, std::vector<float> &stiff_grad,
		std::vector<float> &lambda_grad) const {
	nTarget_grad = vBendN_;
	stiff_grad = vBendStiff_;
	lambda_grad = vBendLambda_;
}

// --- self-collision scan ------------------------------------------------------

void AvbdCpu::uploadSelfCollisionRadii(const float *radii, uint32_t maxNeighborsPerVert) {
	if (!meshReady_) {
		return;
	}
	radii_.assign(radii, radii + nVerts_);
	selfK_ = maxNeighborsPerVert;
	neighbors_.assign(size_t(nVerts_) * selfK_, kSentinel);
}

int AvbdCpu::submitSelfCollisionScan() {
	if (!meshReady_ || radii_.size() != nVerts_ || selfK_ == 0) {
		return -1;
	}
	// The kernel writes each vertex's K sentinels itself before its scan.
	kb_scs::GlobalParams_0 gp{};
	kb_scs::SelfCollisionScanParams_0 sp{ nVerts_, selfK_ };
	gp.params_0 = &sp;
	gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
	gp.radii_0.data = radii_.data(); gp.radii_0.count = nVerts_;
	gp.neighbors_0.data = neighbors_.data(); gp.neighbors_0.count = (uint32_t)neighbors_.size();
	dispatch(nVerts_, &kb_scs::main_0_Thread, &gp);
	return 0;
}

int AvbdCpu::collectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs) {
	out_pairs.clear();
	if (!meshReady_ || radii_.size() != nVerts_ || selfK_ == 0) {
		return -1;
	}
	for (uint32_t i = 0; i < nVerts_; ++i) {
		for (uint32_t k = 0; k < selfK_; ++k) {
			const uint32_t j = neighbors_[size_t(i) * selfK_ + k];
			if (j == kSentinel) {
				break;
			}
			if (j > i) {
				out_pairs.emplace_back(i, j);
			}
		}
	}
	return 0;
}
