// SPDX-License-Identifier: Apache-2.0 OR MIT
#ifndef AVBD_CPU_H
#define AVBD_CPU_H

#include <cstdint>
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
			if (step() != 0) {
				return -1;
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
	uint32_t numColors() const { return uint32_t(colorOffsets_.size()) - 1; }

	void restrictToOwned(uint32_t nOwned);
	void setPositions(const float *positions);

	void readPositions(std::vector<float> &out) const;
	uint32_t nVerts() const { return nVerts_; }
	bool ready() const { return meshReady_; }

private:
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
};

#endif
