// fit_gpu -- the similarity Hessian on the fit worker's local RenderingDevice
// (cut 6g-C, stage 1).
//
// One assembly is one round trip: upload the slots' positions (buffer_update
// before the list), record similarity_hessian_block -> barrier ->
// project_psd12 -> barrier -> csr_gather_df32 into one compute list, submit,
// sync, and read the nonzeros' pairs back through buffer_get_into into the
// pattern's value array. The sync blocks the worker Thread inside its
// vmcall, never the main thread (AGENTS.md rule 4 as scoped by Gate 6G.1);
// the device is created on that Thread by open() and is bound to it, so
// the host keeps one persistent worker for the session (stage_base.gd).
//
// Every RID is permanent across vmcalls (rdc::Device), and the session data
// (hinges, coefficients, weights, the pattern's gather lists) is uploaded
// once by open(). RenderingDevice is reached only through rdc::Device: its
// method-name slots are full (AGENTS), and nothing here needs a new one.
#pragma once

#include "fit_sim_hessian.h"
#include "rd_compute.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fit_gpu {

struct GpuStats {
	int64_t assemblies = 0;   // hook calls answered by the GPU
	int64_t fallbacks = 0;    // hook calls that failed and fell back to the CPU path
	int64_t total_us = 0;     // host time inside assemble(), all calls
	int64_t upload_us = 0;    // of which buffer_update of the positions
	int64_t gpu_us = 0;       // of which list_begin .. sync (record, submit, wait)
	int64_t readback_us = 0;  // of which buffer_get_into + the value fill
	int64_t last_us = 0;
	int64_t gpu_ns = 0;       // GPU timestamps around the list, all calls (0 when off)
	std::string last_error;
};

class GpuHessian {
public:
	// Opens the device on the calling thread, builds the pipelines, uploads
	// the session data. false with *err set when anything is missing (no
	// renderer, a kernel not embedded, a buffer refused).
	bool open(const SimProblem &p, std::string *err);
	// The device alone, on the calling thread: what open() does first, and
	// what the broad phase (fit_broad_phase.h) needs when the Hessian path
	// is off. Idempotent.
	bool open_device(std::string *err);
	bool is_open() const { return open_; }
	// The whole assembly at x into out (= the pattern with values). `sign`
	// scales every block (the gate's flipped-sign control; 1 in the solver).
	bool assemble(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, polyfem::StiffnessMatrix &out);
	// The blocks after the block kernel and (when psd) the projection, read
	// back whole: the gate's per-block comparison.
	bool blocks(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, std::vector<Df> &blocks);
	// Frees every RID and the device. Only from the thread that opened it.
	void close();
	// GPU timestamps around the compute list (two more host calls per round
	// trip); off in the solver, on for the gate's timing.
	void set_timestamps(bool on) { timestamps_ = on; }
	const GpuStats &stats() const { return stats_; }
	rdc::Device &device() { return dev_; }
	std::string device_name();

private:
	bool run(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, bool with_gather, std::string *err);
	bool fail(const std::string &what);

	rdc::Device dev_;
	bool open_ = false;
	bool timestamps_ = false;
	GpuStats stats_;
	std::vector<Df> pos_;
	std::vector<Df> values_;

	::RID sh_block_, sh_psd_, sh_gather_;
	::RID pipe_block_, pipe_psd_, pipe_gather_;
	::RID b_pos_, b_hinge_v_, b_coef_, b_blocks_, b_ptr_, b_src_, b_weight_, b_values_;
	::RID u_block_, u_psd_, u_gather_;
	::RID set_block_, set_psd_, set_gather_;
	uint32_t n_hinges_ = 0;
	uint32_t nnz_ = 0;
	uint32_t psd_last_ = 2;
	float sign_last_ = 0;
};

} // namespace fit_gpu
