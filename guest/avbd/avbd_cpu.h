// SPDX-License-Identifier: Apache-2.0 OR MIT
#ifndef AVBD_CPU_H
#define AVBD_CPU_H

#include <cstdint>
#include <utility>
#include <vector>

class AvbdCpu {
public:
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

	// The rest-shape update on the same topology (the rd solver's twins).
	void updateTriangleRest(const float *invUV, const float *stiffness);
	void updateBendingRest(const float *weight, const float *nTarget, const float *stiffness);
	void updateAttachmentStiffness(const float *stiffness);
	void updateSelfCollisionRadii(const float *radii);

	void setGammaScale(float scale);
	// AVBD Eq. 16 penalty ramp for the dual updates; beta = 0 (default) is
	// the fixed-gamma behaviour.
	void setPenaltyRamp(float beta, float penaltyMax) { beta_ = beta; penaltyMax_ = penaltyMax; }
	void updateState(const float *positions, const float *predicted);
	void updateAttachmentFixedPos(const float *fixedPos);

	int step();
	int stepDualAttachments();
	int stepDualMembrane();
	int stepDualBending();
	// `iters` outer iterations, each followed by the dual updates when `duals`
	// is set. The CPU path has nothing to batch; this keeps the surface equal
	// to AvbdRd so drivers are templated over the backend.
	int run(int iters, bool duals) {
		for (int i = 0; i < iters; ++i) {
			const bool last = i + 1 == iters && dbgColor_ >= 0;
			if (step_impl(last ? dbgColor_ : -1, last ? dbgStage_ : 5) != 0) {
				return -1;
			}
			if (last) {
				break;
			}
			if (duals) {
				stepDualAttachments();
				stepDualMembrane();
				stepDualBending();
			}
		}
		return 0;
	}
	void sync() {}
	// Nothing is ever in flight on the CPU; AvbdRd's is true between a submit
	// and the sync of a later tick (AGENTS.md rule 4).
	bool pending() const { return false; }
	uint32_t numColors() const { return uint32_t(colorOffsets_.size()) - 1; }

	// Reverse-mode adjoint of the last step() (avbd_cpu_backward.cpp).
	// `v_positions_loss` is dL/dx_out, length 3*nVerts. Then the read*Grad
	// accessors, laid out as cloth::AvbdSolver's.
	int stepBackward(const float *v_positions_loss);
	// run(iters, duals) then stepBackward(vOut): the backward pass
	// differentiates the last iteration. One submit on AvbdRd.
	int runWithBackward(int iters, bool duals, const float *v_positions_loss) {
		if (run(iters, duals) != 0) {
			return -1;
		}
		return stepBackward(v_positions_loss);
	}
	void readPositionsGrad(std::vector<float> &out) const;
	void readMassGrad(std::vector<float> &out) const;
	void readPredictedGrad(std::vector<float> &out) const;
	void readSpringGrad(std::vector<float> &restLen_grad, std::vector<float> &stiff_grad) const;
	void readAttachGrad(std::vector<float> &fixedPos_grad, std::vector<float> &stiff_grad,
			std::vector<float> &lambda_grad) const;
	void readTriGrad(std::vector<float> &stiff_grad, std::vector<float> &lambda0_grad,
			std::vector<float> &lambda1_grad) const;
	void readBendGrad(std::vector<float> &nTarget_grad, std::vector<float> &stiff_grad,
			std::vector<float> &lambda_grad) const;

	// Self-collision scan: per-vertex radii, at most K neighbours recorded per
	// vertex; pairs come back once, lower index first.
	// submitSelfCollisionScan runs the scan (on AvbdRd: submits it);
	// collectSelfCollisions reads the pairs back (on AvbdRd: on a later tick).
	// detectSelfCollisions is the two in one call (CPU only: AvbdRd has none,
	// since on the GPU it would sync in its submit's frame, rule 4).
	void uploadSelfCollisionRadii(const float *radii, uint32_t maxNeighborsPerVert);
	int submitSelfCollisionScan();
	int collectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs);
	int detectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs) {
		if (submitSelfCollisionScan() != 0) {
			out_pairs.clear();
			return -1;
		}
		return collectSelfCollisions(out_pairs);
	}

	// Test hook for gradcheck_duals' negative control: false makes the
	// backward bind the live duals instead of the pre-step copies, which is
	// wrong once a dual update has run after the step.
	void setLambdaSnapshotForTest(bool use) { useLambdaSnapshot_ = use; }

	// The per-kernel bisection's hooks, as AvbdRd's (avbd_rd.h): the last
	// iteration of run() stops in colour `color` after `stage` (0 init and the
	// four force kernels, 1-4 the gathers, 5 the solve); -1 runs whole
	// iterations. readDebugForTest reads a buffer by the kernels' name.
	void setDebugStopForTest(int color, int stage) {
		dbgColor_ = color;
		dbgStage_ = stage;
	}
	std::vector<float> readDebugForTest(const char *name) const;

	void restrictToOwned(uint32_t nOwned);
	void setPositions(const float *positions);

	void readPositions(std::vector<float> &out) const;
	uint32_t nVerts() const { return nVerts_; }
	bool ready() const { return meshReady_; }

private:
	// One iteration; colour `stopColor` (-1: none) ends after `stopStage`.
	int step_impl(int stopColor, int stopStage);
	int dbgColor_ = -1, dbgStage_ = -1;
	uint32_t nVerts_ = 0, nSprings_ = 0, nAttach_ = 0, nTri_ = 0, nBend_ = 0;
	float invHSq_ = 0.0f;
	float beta_ = 0.0f, penaltyMax_ = 1e10f;
	bool meshReady_ = false;

	std::vector<float> positions_, predicted_, gScratch_, mass_, hScratch_;
	std::vector<uint32_t> vertPerm_, colorOffsets_{ 0u, 0u };

	std::vector<uint32_t> springP1_, springP2_;
	std::vector<float> springRest_, springStiff_, springGradA_, springHess_;
	std::vector<uint32_t> vSpringOff_, vSpringIdx_, vSpringRole_;

	std::vector<uint32_t> attachVert_;
	std::vector<float> attachFixed_, attachStiff_, attachLambda_, attachGamma_;
	std::vector<float> attachGradV_, attachHess_;
	std::vector<uint32_t> vAttachOff_, vAttachIdx_;

	std::vector<uint32_t> triIdx_;
	std::vector<float> triInvUV_, triStiff_, triLambda0_, triLambda1_, triGamma_;
	std::vector<float> triGrad_, triHess_;
	std::vector<uint32_t> vTriOff_, vTriIdx_, vTriRole_;

	std::vector<uint32_t> bendIdx_;
	std::vector<float> bendWeight_, bendNTarget_, bendStiff_, bendLambda_, bendGamma_;
	std::vector<float> bendGrad_, bendHess_;
	std::vector<uint32_t> vBendOff_, vBendIdx_, vBendRole_;

	// Backward state: the pre-step positions and duals (the last step's
	// force kernels saw these), the identity permutation the scatter runs
	// over (vertPerm_ is shorter after restrictToOwned), and every cotangent.
	bool backwardReady_ = false;
	bool useLambdaSnapshot_ = true;
	std::vector<float> positionsPre_, triLambda0Pre_, triLambda1Pre_, bendLambdaPre_;
	std::vector<uint32_t> identPerm_;
	std::vector<float> vOut_, vG_, vH_, deltaX_, vPosGrad_, vPosInit_, vPosSum_, vPred_, vMass_, hJunk_;
	std::vector<float> vSpringGradA_, vSpringHess_, vSpringPd_, vSpringRest_, vSpringStiff_;
	std::vector<float> vAttachGradV_, vAttachHess_, vAttachFixed_, vAttachStiff_, vAttachLambda_;
	std::vector<float> vTriGrad_, vTriHess_, vTriP_, vTriStiff_, vTriL0_, vTriL1_;
	std::vector<float> vBendGrad_, vBendHess_, vBendP_, vBendN_, vBendStiff_, vBendLambda_;
	// Self-collision.
	std::vector<float> radii_;
	float gammaScale_ = 1.0f; // the product of setGammaScale calls (the update paths re-derive gamma from k)
	std::vector<uint32_t> neighbors_;
	uint32_t selfK_ = 0;
};

#endif
