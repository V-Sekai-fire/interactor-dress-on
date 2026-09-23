// AvbdRd: the reverse-mode adjoint and the self-collision scan on the GPU,
// the same sequence as avbd_cpu_backward.cpp with each kernel a dispatch.
//
// Everything is on the GPU, prologue and epilogue included: deltaX =
// positions - positionsPre and the sum vPosGrad + vPosInit + vOut are saxpby
// dispatches over the 16-byte-strided float3 buffers (4 floats per vertex),
// and the accumulators are zeroed with buffer_clear. The host uploads the
// incoming cotangent and reads nothing back until a later tick (AGENTS.md
// rule 4): stepBackward, runWithBackward and submitSelfCollisionScan submit
// and return; the read* accessors sync only if the submit is still pending.
//
// Barriers follow the data flow: deltaX; solve_apply backward writes v_g/v_H;
// the four gather backwards read them and write disjoint per-family
// cotangents; the force backwards (on the pre-step positions and duals), init
// backward and attachment's 1:1 v_positions write read those and write
// disjoint outputs; the scatter gathers all accumulate into v_positions, so
// they are serialised; the epilogue sums last. The pre-step positions and
// duals come from buffer_copy recorded before the last forward list (see
// record_run()).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "avbd_rd.h"

#include <cstring>

namespace {
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
}

bool AvbdRd::ensure_backward_buffers() {
	if (backwardReady_) {
		return true;
	}
	// The same capacities as the forward buffers (cap_for), so the ragged
	// tails of the unguarded backward kernels stay in bounds.
	const uint32_t nV = nVerts_;
	const uint32_t vcap = cap_for(nV);
	Buf *v3s[] = { &vOut_, &vG_, &deltaX_, &vPosGrad_, &vPosInit_, &vPred_, &vPosSum_, &vPosGradOut_ };
	for (Buf *b : v3s) {
		free_buf(*b);
		*b = make_zero(size_t(16) * vcap);
	}
	free_buf(vH_);
	vH_ = make_zero(size_t(24) * vcap);
	free_buf(hJunk_);
	hJunk_ = make_zero(size_t(24) * vcap);
	free_buf(vMass_);
	vMass_ = make_zero(size_t(4) * vcap);
	Buf *sp[] = { &vSpringGradA_, &vSpringHess_, &vSpringPd_, &vSpringRest_, &vSpringStiff_ };
	for (Buf *b : sp) {
		free_buf(*b);
	}
	if (nSprings_) {
		const uint32_t c = cap_for(nSprings_);
		vSpringGradA_ = make_zero(size_t(16) * c);
		vSpringHess_ = make_zero(size_t(24) * c);
		vSpringPd_ = make_zero(size_t(16) * c);
		vSpringRest_ = make_zero(size_t(4) * c);
		vSpringStiff_ = make_zero(size_t(4) * c);
	}
	Buf *at[] = { &vAttachGradV_, &vAttachHess_, &vAttachFixed_, &vAttachStiff_, &vAttachLambda_ };
	for (Buf *b : at) {
		free_buf(*b);
	}
	if (nAttach_) {
		const uint32_t c = cap_for(nAttach_);
		vAttachGradV_ = make_zero(size_t(16) * c);
		vAttachHess_ = make_zero(size_t(4) * c);
		vAttachFixed_ = make_zero(size_t(16) * c);
		vAttachStiff_ = make_zero(size_t(4) * c);
		vAttachLambda_ = make_zero(size_t(16) * c);
	}
	Buf *tr[] = { &vTriGrad_, &vTriHess_, &vTriP_, &vTriStiff_, &vTriL0_, &vTriL1_ };
	for (Buf *b : tr) {
		free_buf(*b);
	}
	if (nTri_) {
		const uint32_t c = cap_for(nTri_);
		vTriGrad_ = make_zero(size_t(16) * 3 * c);
		vTriHess_ = make_zero(size_t(4) * 3 * c);
		vTriP_ = make_zero(size_t(16) * 3 * c);
		vTriStiff_ = make_zero(size_t(4) * c);
		vTriL0_ = make_zero(size_t(16) * c);
		vTriL1_ = make_zero(size_t(16) * c);
	}
	Buf *bd[] = { &vBendGrad_, &vBendHess_, &vBendP_, &vBendN_, &vBendStiff_, &vBendLambda_ };
	for (Buf *b : bd) {
		free_buf(*b);
	}
	if (nBend_) {
		const uint32_t c = cap_for(nBend_);
		vBendGrad_ = make_zero(size_t(16) * 4 * c);
		vBendHess_ = make_zero(size_t(4) * 4 * c);
		vBendP_ = make_zero(size_t(16) * 4 * c);
		vBendN_ = make_zero(size_t(4) * c);
		vBendStiff_ = make_zero(size_t(4) * c);
		vBendLambda_ = make_zero(size_t(16) * c);
	}
	free_buf(scatterParams_);
	uint32_t gz[4] = { 0u, nV, 0u, 0u };
	scatterParams_.bytes = sizeof gz;
	scatterParams_.rid = d_.uniform_buffer(sizeof gz, gz);
	free_buf(initBwdParams_);
	float ibp[4] = { invHSq_, 0.0f, 0.0f, 0.0f };
	initBwdParams_.bytes = sizeof ibp;
	initBwdParams_.rid = d_.uniform_buffer(sizeof ibp, ibp);
	// saxpby {uint n; float alpha; float beta;}: n counts floats, 4 per
	// 16-byte float3 row.
	struct SaxP {
		uint32_t n;
		float alpha, beta;
		uint32_t pad;
	};
	const SaxP delta{ 4 * nV, 1.0f, -1.0f, 0u };
	const SaxP sum{ 4 * nV, 1.0f, 1.0f, 0u };
	Buf *sx[] = { &sxDeltaParams_, &sxSumParams_, &sxOutParams_ };
	const SaxP *sxv[] = { &delta, &sum, &sum };
	for (int i = 0; i < 3; ++i) {
		free_buf(*sx[i]);
		sx[i]->bytes = sizeof(SaxP);
		sx[i]->rid = d_.uniform_buffer(sizeof(SaxP), sxv[i]);
	}
	backwardReady_ = vOut_.valid() && vH_.valid() && vPosGradOut_.valid() && scatterParams_.valid() &&
			initBwdParams_.valid() && sxOutParams_.valid();
	invalidate_sets();
	if (!backwardReady_) {
		err_ = "backward buffers: " + d_.error();
	}
	return backwardReady_;
}

bool AvbdRd::prepare_backward(const float *v_positions_loss) {
	if (!hasStep_ || !ensure_backward_buffers()) {
		return false;
	}
	// Transfers only, before any compute list: the incoming cotangent up,
	// and the two accumulators the scatter adds into zeroed on the GPU.
	update_v3(vOut_, v_positions_loss, nVerts_);
	d_.buffer_clear(vPosGrad_.rid, 0, vPosGrad_.bytes);
	d_.buffer_clear(hJunk_.rid, 0, hJunk_.bytes);
	return true;
}

void AvbdRd::record_backward() {
	const uint32_t nV = nVerts_;
	// Prologue: deltaX = positions - positionsPre.
	dispatch("saxpby", kSaxDelta, 4 * nV,
			{ { "x", &positions_ }, { "y", &positionsPre_ }, { "dst", &deltaX_ } });
	d_.barrier();
	dispatch("vbd_solve_apply_backward", -1, nV,
			{ { "v_out", &vOut_ }, { "hScratch", &hScratch_ }, { "deltaX", &deltaX_ }, { "v_g", &vG_ },
					{ "v_H", &vH_ } });
	d_.barrier();
	if (nSprings_) {
		dispatch("vbd_gather_spring_backward", -1, nSprings_,
				{ { "springP1Idx", &sp_p1_ }, { "springP2Idx", &sp_p2_ }, { "v_g", &vG_ }, { "v_H", &vH_ },
						{ "v_springGradA", &vSpringGradA_ }, { "v_springHess", &vSpringHess_ } });
	}
	if (nAttach_) {
		dispatch("vbd_gather_attachment_backward", -1, nAttach_,
				{ { "attachVertIdx", &at_vert_ }, { "v_g", &vG_ }, { "v_H", &vH_ },
						{ "v_attachGradV", &vAttachGradV_ }, { "v_attachHessScalar", &vAttachHess_ } });
	}
	if (nTri_) {
		dispatch("vbd_gather_triangle_backward", -1, 3 * nTri_,
				{ { "triIdx", &tri_idx_ }, { "v_g", &vG_ }, { "v_H", &vH_ }, { "v_triGrad", &vTriGrad_ },
						{ "v_triHessScalar", &vTriHess_ } });
	}
	if (nBend_) {
		dispatch("vbd_gather_bending_backward", -1, 4 * nBend_,
				{ { "bendIdx", &bd_idx_ }, { "v_g", &vG_ }, { "v_H", &vH_ }, { "v_bendGrad", &vBendGrad_ },
						{ "v_bendHessScalar", &vBendHess_ } });
	}
	d_.barrier();
	// Force backwards on the pre-step positions and duals, init backward,
	// and the attachment's 1:1 v_positions write: disjoint outputs, one
	// segment.
	if (nAttach_) {
		dispatch("attachment_force_al_backward", -1, nAttach_,
				{ { "positions", &positionsPre_ }, { "vertIdx", &at_vert_ }, { "fixedPos", &at_fixed_ },
						{ "stiffness", &at_k_ }, { "v_gradV", &vAttachGradV_ }, { "v_hessScalar", &vAttachHess_ },
						{ "v_positions", &vPosGrad_ }, { "v_fixedPos", &vAttachFixed_ },
						{ "v_lambda", &vAttachLambda_ }, { "v_stiffness", &vAttachStiff_ } });
	}
	if (nSprings_) {
		dispatch("spring_force_backward", -1, nSprings_,
				{ { "positions", &positionsPre_ }, { "p1Idx", &sp_p1_ }, { "p2Idx", &sp_p2_ },
						{ "restLen", &sp_rest_ }, { "stiffness", &sp_k_ }, { "v_gradA", &vSpringGradA_ },
						{ "v_springHess", &vSpringHess_ }, { "v_p_d", &vSpringPd_ }, { "v_restLen", &vSpringRest_ },
						{ "v_stiffness", &vSpringStiff_ } });
	}
	if (nTri_) {
		dispatch("triangle_membrane_force_al_backward", -1, nTri_,
				{ { "positions", &positionsPre_ }, { "idx", &tri_idx_ }, { "stiffness", &tri_k_ },
						{ "lambda0", useLambdaSnapshot_ ? &tri_l0Pre_ : &tri_l0_ },
						{ "lambda1", useLambdaSnapshot_ ? &tri_l1Pre_ : &tri_l1_ }, { "inv_deltaUV", &tri_invuv_ },
						{ "v_grad", &vTriGrad_ }, { "v_hessScalar", &vTriHess_ }, { "v_p", &vTriP_ },
						{ "v_stiffness", &vTriStiff_ }, { "v_lambda0", &vTriL0_ }, { "v_lambda1", &vTriL1_ } });
	}
	if (nBend_) {
		dispatch("triangle_bending_force_al_backward", -1, nBend_,
				{ { "positions", &positionsPre_ }, { "idx", &bd_idx_ }, { "weight", &bd_w_ }, { "nTarget", &bd_n_ },
						{ "stiffness", &bd_k_ }, { "lambda", useLambdaSnapshot_ ? &bd_lambdaPre_ : &bd_lambda_ }, { "v_grad", &vBendGrad_ },
						{ "v_hessScalar", &vBendHess_ }, { "v_p", &vBendP_ }, { "v_nTarget", &vBendN_ },
						{ "v_stiffness", &vBendStiff_ }, { "v_lambda", &vBendLambda_ } });
	}
	dispatch("vbd_init_backward", -1, nV,
			{ { "positions", &positionsPre_ }, { "predicted", &predicted_ }, { "mass", &mass_ }, { "v_g", &vG_ },
					{ "v_H", &vH_ }, { "v_x", &vPosInit_ }, { "v_y", &vPred_ }, { "v_mass", &vMass_ } });
	d_.barrier();
	// The inertial and direct paths, summed while the scatters run (disjoint
	// buffers): vPosSum = vPosInit + vOut.
	dispatch("saxpby", kSaxSum, 4 * nV,
			{ { "x", &vPosInit_ }, { "y", &vOut_ }, { "dst", &vPosSum_ } });
	// Scatter through the forward gathers over the whole mesh; each
	// accumulates into v_positions, so serialise.
	if (nSprings_) {
		dispatch_scatter("vbd_gather_spring", nV,
				{ { "springGradA", &vSpringPd_ }, { "springHess", &vSpringHess_ }, { "vertSpringOffset", &sp_off_ },
						{ "vertSpringIdx", &sp_idx_ }, { "vertSpringRole", &sp_role_ }, { "gScratch", &vPosGrad_ },
						{ "hScratch", &hJunk_ }, { "vertPerm", &vertPerm_b_ } });
		d_.barrier();
	}
	if (nTri_) {
		dispatch_scatter("vbd_gather_triangle", nV,
				{ { "triGrad", &vTriP_ }, { "triHessScalar", &vTriHess_ }, { "vertTriOffset", &tri_off_ },
						{ "vertTriIdx", &tri_idxc_ }, { "vertTriRole", &tri_role_ }, { "gScratch", &vPosGrad_ },
						{ "hScratch", &hJunk_ }, { "vertPerm", &vertPerm_b_ } });
		d_.barrier();
	}
	if (nBend_) {
		dispatch_scatter("vbd_gather_bending", nV,
				{ { "bendGrad", &vBendP_ }, { "bendHessScalar", &vBendHess_ }, { "vertBendOffset", &bd_off_ },
						{ "vertBendIdx", &bd_idxc_ }, { "vertBendRole", &bd_role_ }, { "gScratch", &vPosGrad_ },
						{ "hScratch", &hJunk_ }, { "vertPerm", &vertPerm_b_ } });
		d_.barrier();
	}
	if (!nSprings_ && !nTri_ && !nBend_) {
		d_.barrier(); // vPosSum before the epilogue
	}
	// Epilogue: vPosGradOut = vPosGrad + vPosSum.
	dispatch("saxpby", kSaxOut, 4 * nV,
			{ { "x", &vPosGrad_ }, { "y", &vPosSum_ }, { "dst", &vPosGradOut_ } });
}

int AvbdRd::stepBackward(const float *v_positions_loss) {
	if (!prologue()) {
		return -1;
	}
	if (!prepare_backward(v_positions_loss)) {
		return -1;
	}
	d_.list_begin();
	record_backward();
	end_record_and_submit();
	return err_.empty() ? 0 : -1;
}

int AvbdRd::runWithBackward(int iters, bool duals, const float *v_positions_loss) {
	if (iters < 1 || !prologue()) {
		return -1;
	}
	// The cotangent upload and clears touch no forward buffer, so they go
	// first; the forward's last list stays open and the backward follows it
	// after a barrier (record_iteration and record_duals end with one).
	hasStep_ = true; // record_run snapshots before the last iteration
	if (!prepare_backward(v_positions_loss)) {
		return -1;
	}
	record_run(iters, duals);
	record_backward();
	end_record_and_submit();
	return err_.empty() ? 0 : -1;
}

std::vector<float> AvbdRd::read_v3(const Buf &b, uint32_t n) {
	std::vector<float> out(size_t(3) * n, 0.0f);
	if (!b.valid()) {
		return out;
	}
	if (pending_) {
		sync();
	}
	const std::vector<uint8_t> bytes = d_.buffer_get(b.rid);
	if (bytes.size() < size_t(16) * n) {
		return out;
	}
	const float *f = reinterpret_cast<const float *>(bytes.data());
	for (uint32_t i = 0; i < n; ++i) {
		out[3 * i + 0] = f[4 * i + 0];
		out[3 * i + 1] = f[4 * i + 1];
		out[3 * i + 2] = f[4 * i + 2];
	}
	return out;
}

std::vector<float> AvbdRd::read_f32(const Buf &b, uint32_t n) {
	std::vector<float> out(n, 0.0f);
	if (!b.valid() || n == 0) {
		return out;
	}
	if (pending_) {
		sync();
	}
	const std::vector<uint8_t> bytes = d_.buffer_get(b.rid);
	if (bytes.size() < size_t(4) * n) {
		return out;
	}
	std::memcpy(out.data(), bytes.data(), size_t(4) * n);
	return out;
}

void AvbdRd::readPositionsGrad(std::vector<float> &out) { out = read_v3(vPosGradOut_, nVerts_); }
void AvbdRd::readMassGrad(std::vector<float> &out) { out = read_f32(vMass_, nVerts_); }
void AvbdRd::readPredictedGrad(std::vector<float> &out) { out = read_v3(vPred_, nVerts_); }
void AvbdRd::readSpringGrad(std::vector<float> &restLen_grad, std::vector<float> &stiff_grad) {
	restLen_grad = read_f32(vSpringRest_, nSprings_);
	stiff_grad = read_f32(vSpringStiff_, nSprings_);
}
void AvbdRd::readAttachGrad(std::vector<float> &fixedPos_grad, std::vector<float> &stiff_grad,
		std::vector<float> &lambda_grad) {
	fixedPos_grad = read_v3(vAttachFixed_, nAttach_);
	stiff_grad = read_f32(vAttachStiff_, nAttach_);
	lambda_grad = read_v3(vAttachLambda_, nAttach_);
}
void AvbdRd::readTriGrad(std::vector<float> &stiff_grad, std::vector<float> &lambda0_grad,
		std::vector<float> &lambda1_grad) {
	stiff_grad = read_f32(vTriStiff_, nTri_);
	lambda0_grad = read_v3(vTriL0_, nTri_);
	lambda1_grad = read_v3(vTriL1_, nTri_);
}
void AvbdRd::readBendGrad(std::vector<float> &nTarget_grad, std::vector<float> &stiff_grad,
		std::vector<float> &lambda_grad) {
	nTarget_grad = read_f32(vBendN_, nBend_);
	stiff_grad = read_f32(vBendStiff_, nBend_);
	lambda_grad = read_v3(vBendLambda_, nBend_);
}

// --- self-collision scan ------------------------------------------------------

void AvbdRd::uploadSelfCollisionRadii(const float *radii, uint32_t maxNeighborsPerVert) {
	if (!meshReady_ || maxNeighborsPerVert == 0) {
		return;
	}
	if (pending_) {
		sync();
	}
	selfK_ = maxNeighborsPerVert;
	free_buf(radii_);
	free_buf(neighbors_);
	free_buf(selfParams_);
	radii_ = make_f32(std::vector<float>(radii, radii + nVerts_));
	// The kernel writes each vertex's K sentinels itself before its scan;
	// the buffer only has to exist (created with contents, AGENTS.md).
	neighbors_ = make_zero(size_t(4) * nVerts_ * selfK_);
	uint32_t sp[4] = { nVerts_, selfK_, 0u, 0u };
	selfParams_.bytes = sizeof sp;
	selfParams_.rid = d_.uniform_buffer(sizeof sp, sp);
	invalidate_sets();
}

int AvbdRd::submitSelfCollisionScan() {
	if (!radii_.valid() || selfK_ == 0 || !prologue()) {
		return -1;
	}
	d_.list_begin();
	dispatch("self_collision_scan", -1, nVerts_,
			{ { "positions", &positions_ }, { "radii", &radii_ }, { "neighbors", &neighbors_ } });
	end_record_and_submit();
	return err_.empty() ? 0 : -1;
}

int AvbdRd::collectSelfCollisions(std::vector<std::pair<uint32_t, uint32_t>> &out_pairs) {
	out_pairs.clear();
	if (!meshReady_ || !neighbors_.valid() || selfK_ == 0) {
		return -1;
	}
	if (pending_) {
		sync();
	}
	const std::vector<uint8_t> bytes = d_.buffer_get(neighbors_.rid);
	if (bytes.size() < size_t(4) * nVerts_ * selfK_) {
		return -1;
	}
	const uint32_t *nb = reinterpret_cast<const uint32_t *>(bytes.data());
	for (uint32_t i = 0; i < nVerts_; ++i) {
		for (uint32_t k = 0; k < selfK_; ++k) {
			const uint32_t j = nb[size_t(i) * selfK_ + k];
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
