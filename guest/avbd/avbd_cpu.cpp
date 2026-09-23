// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "avbd_cpu.h"

#include <cstring>
#include <string>

#include "avbd_topology.h"
#include "slang-cpp-prelude.h"

// Each slangc emit puts main_0 / GlobalParams_0 at file scope under
// extern "C". Redefining the EXTERN_C macros empty demotes those symbols to
// ordinary C++, so each emit included in its own namespace stays isolated
// and all thirteen kernels coexist in one TU (the cg_demo_test pattern).
#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace k_init {
#include "vbd_init_emit.cpp"
}
namespace k_sf {
#include "spring_force_emit.cpp"
}
namespace k_gs {
#include "vbd_gather_spring_emit.cpp"
}
namespace k_afa {
#include "attachment_force_al_emit.cpp"
}
namespace k_ga {
#include "vbd_gather_attachment_emit.cpp"
}
namespace k_tmf {
#include "triangle_membrane_force_al_emit.cpp"
}
namespace k_gt {
#include "vbd_gather_triangle_emit.cpp"
}
namespace k_tbf {
#include "triangle_bending_force_al_emit.cpp"
}
namespace k_gb {
#include "vbd_gather_bending_emit.cpp"
}
namespace k_sa {
#include "vbd_solve_apply_emit.cpp"
}
namespace k_adu {
#include "attachment_dual_update_emit.cpp"
}
namespace k_tmd {
#include "triangle_membrane_dual_update_emit.cpp"
}
namespace k_tbd {
#include "triangle_bending_dual_update_emit.cpp"
}

namespace {

using Vec3 = Vector<float, 3>;
using ThreadFn = void (*)(ComputeThreadVaryingInput *, void *, void *);

// Metal dispatchThreads(count) runs exactly count threads with global id
// 0..count-1; main_0_Thread runs one thread at the id we hand it.
inline void dispatch(uint32_t count, ThreadFn fn, void *gp) {
	for (uint32_t lane = 0; lane < count; ++lane) {
		ComputeThreadVaryingInput t{};
		t.groupID = uint3(0u, 0u, 0u);
		t.groupThreadID = uint3(lane, 0u, 0u);
		fn(&t, nullptr, gp);
	}
}

inline Vec3 *v3(std::vector<float> &b) { return reinterpret_cast<Vec3 *>(b.data()); }
inline const Vec3 *v3(const std::vector<float> &b) {
	return reinterpret_cast<const Vec3 *>(b.data());
}

} // namespace

void AvbdCpu::setupMesh(uint32_t nVerts, const float *positions, const float *predicted,
		const float *mass, float invHSquared) {
	nVerts_ = nVerts;
	invHSq_ = invHSquared;
	positions_.assign(positions, positions + 3 * nVerts);
	predicted_.assign(predicted, predicted + 3 * nVerts);
	mass_.assign(mass, mass + nVerts);
	gScratch_.assign(3 * nVerts, 0.0f);
	hScratch_.assign(6 * nVerts, 0.0f);
	vertPerm_.resize(nVerts);
	for (uint32_t i = 0; i < nVerts; ++i) {
		vertPerm_[i] = i;
	}
	identPerm_ = vertPerm_;
	colorOffsets_ = { 0u, nVerts };
	// A new mesh invalidates the last step and the self-collision setup.
	positionsPre_.clear();
	backwardReady_ = false;
	radii_.clear();
	neighbors_.clear();
	selfK_ = 0;
	meshReady_ = true;
}

void AvbdCpu::uploadSprings(uint32_t nSprings, const uint32_t *p1Idx, const uint32_t *p2Idx,
		const float *restLen, const float *stiffness) {
	nSprings_ = nSprings;
	springP1_.assign(p1Idx, p1Idx + nSprings);
	springP2_.assign(p2Idx, p2Idx + nSprings);
	springRest_.assign(restLen, restLen + nSprings);
	springStiff_.assign(stiffness, stiffness + nSprings);
	springGradA_.assign(3 * nSprings, 0.0f);
	springHess_.assign(6 * nSprings, 0.0f);
	std::vector<uint32_t> ends(2 * nSprings);
	for (uint32_t i = 0; i < nSprings; ++i) {
		ends[2 * i] = p1Idx[i];
		ends[2 * i + 1] = p2Idx[i];
	}
	avbd::build_csr(nVerts_, nSprings, 2, ends.data(), vSpringOff_, vSpringIdx_, vSpringRole_);
}

void AvbdCpu::uploadAttachments(uint32_t nAttach, const uint32_t *vertIdx,
		const float *fixedPos, const float *stiffness) {
	nAttach_ = nAttach;
	attachVert_.assign(vertIdx, vertIdx + nAttach);
	attachFixed_.assign(fixedPos, fixedPos + 3 * nAttach);
	attachStiff_.assign(stiffness, stiffness + nAttach);
	attachGamma_.assign(stiffness, stiffness + nAttach);
	attachLambda_.assign(3 * nAttach, 0.0f);
	attachGradV_.assign(3 * nAttach, 0.0f);
	attachHess_.assign(nAttach, 0.0f);
	std::vector<uint32_t> roleUnused;
	avbd::build_csr(nVerts_, nAttach, 1, vertIdx, vAttachOff_, vAttachIdx_, roleUnused);
}

void AvbdCpu::uploadTriangles(uint32_t nTri, const uint32_t *triIdx, const float *invUV,
		const float *stiffness) {
	nTri_ = nTri;
	triIdx_.assign(triIdx, triIdx + 3 * nTri);
	triInvUV_.assign(invUV, invUV + 4 * nTri);
	triStiff_.assign(stiffness, stiffness + nTri);
	triGamma_.assign(stiffness, stiffness + nTri);
	triLambda0_.assign(3 * nTri, 0.0f);
	triLambda1_.assign(3 * nTri, 0.0f);
	triLambda0Pre_ = triLambda0_;
	triLambda1Pre_ = triLambda1_;
	triGrad_.assign(3 * 3 * nTri, 0.0f);
	triHess_.assign(3 * nTri, 0.0f);
	avbd::build_csr(nVerts_, nTri, 3, triIdx, vTriOff_, vTriIdx_, vTriRole_);
}

void AvbdCpu::uploadBendings(uint32_t nBend, const uint32_t *bendIdx, const float *weight,
		const float *nTarget, const float *stiffness) {
	nBend_ = nBend;
	bendIdx_.assign(bendIdx, bendIdx + 4 * nBend);
	bendWeight_.assign(weight, weight + 4 * nBend);
	bendNTarget_.assign(nTarget, nTarget + nBend);
	bendStiff_.assign(stiffness, stiffness + nBend);
	bendGamma_.assign(stiffness, stiffness + nBend);
	bendLambda_.assign(3 * nBend, 0.0f);
	bendLambdaPre_ = bendLambda_;
	bendGrad_.assign(3 * 4 * nBend, 0.0f);
	bendHess_.assign(4 * nBend, 0.0f);
	avbd::build_csr(nVerts_, nBend, 4, bendIdx, vBendOff_, vBendIdx_, vBendRole_);
}

void AvbdCpu::buildColoring() {
	avbd::greedy_coloring(nVerts_, nSprings_, springP1_.data(), springP2_.data(), nTri_, triIdx_.data(),
			nBend_, bendIdx_.data(), vertPerm_, colorOffsets_);
}

void AvbdCpu::setGammaScale(float scale) {
	gammaScale_ *= scale;
	for (float &g : attachGamma_) {
		g *= scale;
	}
	for (float &g : triGamma_) {
		g *= scale;
	}
	for (float &g : bendGamma_) {
		g *= scale;
	}
}

void AvbdCpu::updateTriangleRest(const float *invUV, const float *stiffness) {
	if (nTri_ == 0) {
		return;
	}
	triInvUV_.assign(invUV, invUV + 4 * size_t(nTri_));
	triStiff_.assign(stiffness, stiffness + nTri_);
	triGamma_.assign(stiffness, stiffness + nTri_);
	for (float &g : triGamma_) {
		g *= gammaScale_;
	}
}

void AvbdCpu::updateBendingRest(const float *weight, const float *nTarget, const float *stiffness) {
	if (nBend_ == 0) {
		return;
	}
	bendWeight_.assign(weight, weight + 4 * size_t(nBend_));
	bendNTarget_.assign(nTarget, nTarget + nBend_);
	bendStiff_.assign(stiffness, stiffness + nBend_);
	bendGamma_.assign(stiffness, stiffness + nBend_);
	for (float &g : bendGamma_) {
		g *= gammaScale_;
	}
}

void AvbdCpu::updateAttachmentStiffness(const float *stiffness) {
	if (nAttach_ == 0) {
		return;
	}
	attachStiff_.assign(stiffness, stiffness + nAttach_);
	attachGamma_.assign(stiffness, stiffness + nAttach_);
	for (float &g : attachGamma_) {
		g *= gammaScale_;
	}
}

void AvbdCpu::updateSelfCollisionRadii(const float *radii) {
	if (!meshReady_ || radii_.size() != nVerts_) {
		return;
	}
	radii_.assign(radii, radii + nVerts_);
}

void AvbdCpu::updateState(const float *positions, const float *predicted) {
	positions_.assign(positions, positions + 3 * nVerts_);
	predicted_.assign(predicted, predicted + 3 * nVerts_);
}

void AvbdCpu::updateAttachmentFixedPos(const float *fixedPos) {
	if (nAttach_ == 0) {
		return;
	}
	attachFixed_.assign(fixedPos, fixedPos + 3 * nAttach_);
}

int AvbdCpu::step() {
	return step_impl(-1, 5);
}

int AvbdCpu::step_impl(int stopColor, int stopStage) {
	if (!meshReady_) {
		return -1;
	}
	// The backward pass differentiates this step: keep the positions and
	// duals its force kernels see (the dual updates run after it).
	positionsPre_ = positions_;
	triLambda0Pre_ = triLambda0_;
	triLambda1Pre_ = triLambda1_;
	bendLambdaPre_ = bendLambda_;
	const uint32_t numColors = static_cast<uint32_t>(colorOffsets_.size()) - 1;
	for (uint32_t k = 0; k < numColors; ++k) {
		const uint32_t offset = colorOffsets_[k];
		const uint32_t count = colorOffsets_[k + 1] - offset;
		if (stopColor >= 0 && int(k) > stopColor) {
			break;
		}
		const int upTo = int(k) == stopColor ? stopStage : 5;
		if (count == 0) {
			continue;
		}

		{
			k_init::GlobalParams_0 gp{};
			k_init::VbdInitParams_0 ip{ invHSq_, offset, count };
			gp.params_0 = &ip;
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.predicted_0.data = v3(predicted_); gp.predicted_0.count = nVerts_;
			gp.mass_0.data = mass_.data(); gp.mass_0.count = nVerts_;
			gp.gScratch_0.data = v3(gScratch_); gp.gScratch_0.count = nVerts_;
			gp.hScratch_0.data = hScratch_.data(); gp.hScratch_0.count = 6 * nVerts_;
			gp.vertPerm_0.data = vertPerm_.data(); gp.vertPerm_0.count = nVerts_;
			dispatch(count, &k_init::main_0_Thread, &gp);
		}

		if (nSprings_ > 0) {
			k_sf::GlobalParams_0 gp{};
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.p1Idx_0.data = springP1_.data(); gp.p1Idx_0.count = nSprings_;
			gp.p2Idx_0.data = springP2_.data(); gp.p2Idx_0.count = nSprings_;
			gp.restLen_0.data = springRest_.data(); gp.restLen_0.count = nSprings_;
			gp.stiffness_0.data = springStiff_.data(); gp.stiffness_0.count = nSprings_;
			gp.gradA_0.data = v3(springGradA_); gp.gradA_0.count = nSprings_;
			gp.hess_0.data = springHess_.data(); gp.hess_0.count = 6 * nSprings_;
			dispatch(nSprings_, &k_sf::main_0_Thread, &gp);
		}
		if (nAttach_ > 0) {
			k_afa::GlobalParams_0 gp{};
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.vertIdx_0.data = attachVert_.data(); gp.vertIdx_0.count = nAttach_;
			gp.fixedPos_0.data = v3(attachFixed_); gp.fixedPos_0.count = nAttach_;
			gp.stiffness_0.data = attachStiff_.data(); gp.stiffness_0.count = nAttach_;
			gp.lambda_0.data = v3(attachLambda_); gp.lambda_0.count = nAttach_;
			gp.gradV_0.data = v3(attachGradV_); gp.gradV_0.count = nAttach_;
			gp.hessScalar_0.data = attachHess_.data(); gp.hessScalar_0.count = nAttach_;
			dispatch(nAttach_, &k_afa::main_0_Thread, &gp);
		}
		if (nTri_ > 0) {
			k_tmf::GlobalParams_0 gp{};
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.idx_0.data = triIdx_.data(); gp.idx_0.count = 3 * nTri_;
			gp.stiffness_0.data = triStiff_.data(); gp.stiffness_0.count = nTri_;
			gp.lambda0_0.data = v3(triLambda0_); gp.lambda0_0.count = nTri_;
			gp.lambda1_0.data = v3(triLambda1_); gp.lambda1_0.count = nTri_;
			gp.grad_0.data = v3(triGrad_); gp.grad_0.count = 3 * nTri_;
			gp.hessScalar_0.data = triHess_.data(); gp.hessScalar_0.count = 3 * nTri_;
			gp.inv_deltaUV_0.data = triInvUV_.data(); gp.inv_deltaUV_0.count = 4 * nTri_;
			dispatch(nTri_, &k_tmf::main_0_Thread, &gp);
		}
		if (nBend_ > 0) {
			k_tbf::GlobalParams_0 gp{};
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.idx_0.data = bendIdx_.data(); gp.idx_0.count = 4 * nBend_;
			gp.weight_0.data = bendWeight_.data(); gp.weight_0.count = 4 * nBend_;
			gp.nTarget_0.data = bendNTarget_.data(); gp.nTarget_0.count = nBend_;
			gp.stiffness_0.data = bendStiff_.data(); gp.stiffness_0.count = nBend_;
			gp.lambda_0.data = v3(bendLambda_); gp.lambda_0.count = nBend_;
			gp.grad_0.data = v3(bendGrad_); gp.grad_0.count = 4 * nBend_;
			gp.hessScalar_0.data = bendHess_.data(); gp.hessScalar_0.count = 4 * nBend_;
			dispatch(nBend_, &k_tbf::main_0_Thread, &gp);
		}
		// The gathers after every force kernel: forces read positions only, so
		// this order is the interleaved one's arithmetic, and it is AvbdRd's.
		if (upTo < 1) {
			break;
		}
		if (nSprings_ > 0) {
			k_gs::GlobalParams_0 gg{};
			k_gs::VbdGatherSpringParams_0 gpar{ offset, count };
			gg.springGradA_0.data = v3(springGradA_); gg.springGradA_0.count = nSprings_;
			gg.springHess_0.data = springHess_.data(); gg.springHess_0.count = 6 * nSprings_;
			gg.vertSpringOffset_0.data = vSpringOff_.data(); gg.vertSpringOffset_0.count = nVerts_ + 1;
			gg.vertSpringIdx_0.data = vSpringIdx_.data(); gg.vertSpringIdx_0.count = (uint32_t)vSpringIdx_.size();
			gg.vertSpringRole_0.data = vSpringRole_.data(); gg.vertSpringRole_0.count = (uint32_t)vSpringRole_.size();
			gg.gScratch_0.data = v3(gScratch_); gg.gScratch_0.count = nVerts_;
			gg.hScratch_0.data = hScratch_.data(); gg.hScratch_0.count = 6 * nVerts_;
			gg.vertPerm_0.data = vertPerm_.data(); gg.vertPerm_0.count = nVerts_;
			gg.params_0 = &gpar;
			dispatch(count, &k_gs::main_0_Thread, &gg);
		}
		if (upTo < 2) {
			break;
		}
		if (nAttach_ > 0) {
			k_ga::GlobalParams_0 gg{};
			k_ga::VbdGatherAttachmentParams_0 gpar{ offset, count };
			gg.attachGradV_0.data = v3(attachGradV_); gg.attachGradV_0.count = nAttach_;
			gg.attachHessScalar_0.data = attachHess_.data(); gg.attachHessScalar_0.count = nAttach_;
			gg.vertAttachOffset_0.data = vAttachOff_.data(); gg.vertAttachOffset_0.count = nVerts_ + 1;
			gg.vertAttachIdx_0.data = vAttachIdx_.data(); gg.vertAttachIdx_0.count = (uint32_t)vAttachIdx_.size();
			gg.gScratch_0.data = v3(gScratch_); gg.gScratch_0.count = nVerts_;
			gg.hScratch_0.data = hScratch_.data(); gg.hScratch_0.count = 6 * nVerts_;
			gg.vertPerm_0.data = vertPerm_.data(); gg.vertPerm_0.count = nVerts_;
			gg.params_0 = &gpar;
			dispatch(count, &k_ga::main_0_Thread, &gg);
		}
		if (upTo < 3) {
			break;
		}
		if (nTri_ > 0) {
			k_gt::GlobalParams_0 gg{};
			k_gt::VbdGatherTriangleParams_0 gpar{ offset, count };
			gg.triGrad_0.data = v3(triGrad_); gg.triGrad_0.count = 3 * nTri_;
			gg.triHessScalar_0.data = triHess_.data(); gg.triHessScalar_0.count = 3 * nTri_;
			gg.vertTriOffset_0.data = vTriOff_.data(); gg.vertTriOffset_0.count = nVerts_ + 1;
			gg.vertTriIdx_0.data = vTriIdx_.data(); gg.vertTriIdx_0.count = (uint32_t)vTriIdx_.size();
			gg.vertTriRole_0.data = vTriRole_.data(); gg.vertTriRole_0.count = (uint32_t)vTriRole_.size();
			gg.gScratch_0.data = v3(gScratch_); gg.gScratch_0.count = nVerts_;
			gg.hScratch_0.data = hScratch_.data(); gg.hScratch_0.count = 6 * nVerts_;
			gg.vertPerm_0.data = vertPerm_.data(); gg.vertPerm_0.count = nVerts_;
			gg.params_0 = &gpar;
			dispatch(count, &k_gt::main_0_Thread, &gg);
		}
		if (upTo < 4) {
			break;
		}
		if (nBend_ > 0) {
			k_gb::GlobalParams_0 gg{};
			k_gb::VbdGatherBendingParams_0 gpar{ offset, count };
			gg.bendGrad_0.data = v3(bendGrad_); gg.bendGrad_0.count = 4 * nBend_;
			gg.bendHessScalar_0.data = bendHess_.data(); gg.bendHessScalar_0.count = 4 * nBend_;
			gg.vertBendOffset_0.data = vBendOff_.data(); gg.vertBendOffset_0.count = nVerts_ + 1;
			gg.vertBendIdx_0.data = vBendIdx_.data(); gg.vertBendIdx_0.count = (uint32_t)vBendIdx_.size();
			gg.vertBendRole_0.data = vBendRole_.data(); gg.vertBendRole_0.count = (uint32_t)vBendRole_.size();
			gg.gScratch_0.data = v3(gScratch_); gg.gScratch_0.count = nVerts_;
			gg.hScratch_0.data = hScratch_.data(); gg.hScratch_0.count = 6 * nVerts_;
			gg.vertPerm_0.data = vertPerm_.data(); gg.vertPerm_0.count = nVerts_;
			gg.params_0 = &gpar;
			dispatch(count, &k_gb::main_0_Thread, &gg);
		}
		if (upTo < 5) {
			break;
		}

		{
			k_sa::GlobalParams_0 gp{};
			k_sa::VbdSolveApplyParams_0 sp{ offset, count };
			gp.gScratch_0.data = v3(gScratch_); gp.gScratch_0.count = nVerts_;
			gp.hScratch_0.data = hScratch_.data(); gp.hScratch_0.count = 6 * nVerts_;
			gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
			gp.vertPerm_0.data = vertPerm_.data(); gp.vertPerm_0.count = nVerts_;
			gp.params_0 = &sp;
			dispatch(count, &k_sa::main_0_Thread, &gp);
		}
	}
	return 0;
}

int AvbdCpu::stepDualAttachments() {
	if (!meshReady_) {
		return -1;
	}
	if (nAttach_ == 0) {
		return 0;
	}
	k_adu::GlobalParams_0 gp{};
	k_adu::AttachmentDualUpdateParams_0 dp{ beta_, penaltyMax_, nAttach_ };
	gp.params_0 = &dp;
	gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
	gp.vertIdx_0.data = attachVert_.data(); gp.vertIdx_0.count = nAttach_;
	gp.fixedPos_0.data = v3(attachFixed_); gp.fixedPos_0.count = nAttach_;
	gp.gamma_0.data = attachGamma_.data(); gp.gamma_0.count = nAttach_;
	gp.lambda_0.data = v3(attachLambda_); gp.lambda_0.count = nAttach_;
	dispatch(nAttach_, &k_adu::main_0_Thread, &gp);
	return 0;
}

int AvbdCpu::stepDualMembrane() {
	if (!meshReady_) {
		return -1;
	}
	if (nTri_ == 0) {
		return 0;
	}
	k_tmd::GlobalParams_0 gp{};
	k_tmd::TriangleMembraneDualUpdateParams_0 dp{ beta_, penaltyMax_, nTri_ };
	gp.params_0 = &dp;
	gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
	gp.idx_0.data = triIdx_.data(); gp.idx_0.count = 3 * nTri_;
	gp.gamma_0.data = triGamma_.data(); gp.gamma_0.count = nTri_;
	gp.lambda0_0.data = v3(triLambda0_); gp.lambda0_0.count = nTri_;
	gp.lambda1_0.data = v3(triLambda1_); gp.lambda1_0.count = nTri_;
	gp.inv_deltaUV_0.data = triInvUV_.data(); gp.inv_deltaUV_0.count = 4 * nTri_;
	dispatch(nTri_, &k_tmd::main_0_Thread, &gp);
	return 0;
}

int AvbdCpu::stepDualBending() {
	if (!meshReady_) {
		return -1;
	}
	if (nBend_ == 0) {
		return 0;
	}
	k_tbd::GlobalParams_0 gp{};
	k_tbd::TriangleBendingDualUpdateParams_0 dp{ beta_, penaltyMax_, nBend_ };
	gp.params_0 = &dp;
	gp.positions_0.data = v3(positions_); gp.positions_0.count = nVerts_;
	gp.idx_0.data = bendIdx_.data(); gp.idx_0.count = 4 * nBend_;
	gp.weight_0.data = bendWeight_.data(); gp.weight_0.count = 4 * nBend_;
	gp.nTarget_0.data = bendNTarget_.data(); gp.nTarget_0.count = nBend_;
	gp.gamma_0.data = bendGamma_.data(); gp.gamma_0.count = nBend_;
	gp.lambda_0.data = v3(bendLambda_); gp.lambda_0.count = nBend_;
	dispatch(nBend_, &k_tbd::main_0_Thread, &gp);
	return 0;
}

void AvbdCpu::restrictToOwned(uint32_t nOwned) {
	if (nOwned > nVerts_) {
		nOwned = nVerts_;
	}
	vertPerm_.resize(nOwned);
	for (uint32_t i = 0; i < nOwned; ++i) {
		vertPerm_[i] = i;
	}
	colorOffsets_ = { 0u, nOwned };
}

void AvbdCpu::setPositions(const float *positions) {
	positions_.assign(positions, positions + 3 * nVerts_);
}

void AvbdCpu::readPositions(std::vector<float> &out) const {
	out = positions_;
}

std::vector<float> AvbdCpu::readDebugForTest(const char *name) const {
	const std::string n = name;
	if (n == "positions") return positions_;
	if (n == "gScratch") return gScratch_;
	if (n == "hScratch") return hScratch_;
	if (n == "attachGradV") return attachGradV_;
	if (n == "attachHess") return attachHess_;
	if (n == "triGrad") return triGrad_;
	if (n == "triHess") return triHess_;
	if (n == "bendGrad") return bendGrad_;
	if (n == "bendHess") return bendHess_;
	return {};
}
