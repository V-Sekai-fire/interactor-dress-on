// fit_gpu -- the similarity Hessian on the fit worker's RenderingDevice. See fit_gpu.h.
#include "fit_gpu.h"

#include "rd_enums.h"

#include "fit_kernels.inc" // kernels/fit/gen.sh: namespace fit_kernels, SPIR-V by name

#include <cstring>

namespace fit_gpu {

namespace {

struct BlockParams {
	uint32_t count;
	float sign;
	uint32_t pad[2];
};
struct PsdParams {
	uint32_t count;
	uint32_t enabled;
	uint32_t pad[2];
};
struct GatherParams {
	uint32_t nnz;
	uint32_t pad[3];
};

const uint8_t *kernel(const char *name, size_t &bytes) {
	const fit_kernels::Entry *e = fit_kernels::find(name);
	if (!e) {
		bytes = 0;
		return nullptr;
	}
	bytes = e->size;
	return e->bytes;
}

} // namespace

bool GpuHessian::fail(const std::string &what) {
	stats_.last_error = what + (dev_.error().empty() ? "" : " (" + dev_.error() + ")");
	return false;
}

std::string GpuHessian::device_name() {
	return dev_.ok() ? dev_.device_name() : std::string("(no device)");
}

bool GpuHessian::open_device(std::string *err) {
	if (dev_.ok())
		return true;
	if (!dev_.open()) {
		if (err)
			*err = dev_.error();
		return false;
	}
	dev_.set_worker(true);
	dev_.set_recovery(true);
	return true;
}

bool GpuHessian::open(const SimProblem &p, std::string *err) {
	if (open_)
		return true;
	if (p.n_hinges <= 0 || p.nnz() == 0) {
		if (err)
			*err = "no hinges";
		return false;
	}
	if (!open_device(err))
		return false;
	auto bail = [&](const std::string &what) {
		if (err)
			*err = what + (dev_.error().empty() ? "" : ": " + dev_.error());
		close();
		return false;
	};
	struct K {
		const char *name;
		::RID *shader, *pipe;
	} ks[3] = { { "similarity_hessian_block", &sh_block_, &pipe_block_ },
		{ "project_psd12", &sh_psd_, &pipe_psd_ },
		{ "csr_gather_df32", &sh_gather_, &pipe_gather_ } };
	for (const K &k : ks) {
		size_t bytes = 0;
		const uint8_t *spv = kernel(k.name, bytes);
		if (!spv)
			return bail(std::string(k.name) + " is not embedded");
		*k.shader = dev_.shader_from_spirv(spv, bytes);
		if (!k.shader->index)
			return bail(std::string(k.name) + ": shader");
		*k.pipe = dev_.compute_pipeline(*k.shader);
		if (!k.pipe->index)
			return bail(std::string(k.name) + ": pipeline");
	}
	n_hinges_ = uint32_t(p.n_hinges);
	nnz_ = uint32_t(p.nnz());
	pos_.assign(size_t(p.n_slots) * 3, Df{});
	values_.assign(p.nnz(), Df{});
	// Session data once; positions and values per assembly.
	b_pos_ = dev_.storage_buffer(pos_.size() * sizeof(Df), pos_.data());
	b_hinge_v_ = dev_.storage_buffer(p.hinge_v.size() * 4, p.hinge_v.data());
	b_coef_ = dev_.storage_buffer(p.coef.size() * sizeof(Df), p.coef.data());
	b_blocks_ = dev_.storage_buffer_empty(size_t(p.n_hinges) * kBlock * sizeof(Df));
	b_ptr_ = dev_.storage_buffer(p.ptr.size() * 4, p.ptr.data());
	b_src_ = dev_.storage_buffer(p.src.size() * 4, p.src.data());
	b_weight_ = dev_.storage_buffer(p.weight.size() * sizeof(Df), p.weight.data());
	b_values_ = dev_.storage_buffer_empty(p.nnz() * sizeof(Df));
	const BlockParams bp{ n_hinges_, 1.0f, { 0, 0 } };
	const PsdParams pp{ n_hinges_, 1u, { 0, 0 } };
	const GatherParams gp{ nnz_, { 0, 0, 0 } };
	u_block_ = dev_.uniform_buffer(sizeof bp, &bp);
	u_psd_ = dev_.uniform_buffer(sizeof pp, &pp);
	u_gather_ = dev_.uniform_buffer(sizeof gp, &gp);
	for (::RID *r : { &b_pos_, &b_hinge_v_, &b_coef_, &b_blocks_, &b_ptr_, &b_src_, &b_weight_, &b_values_, &u_block_, &u_psd_, &u_gather_ })
		if (!r->index)
			return bail("buffers");
	const int U = rdc::UNIFORM_TYPE_UNIFORM_BUFFER, S = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
	set_block_ = dev_.uniform_set(sh_block_, { { 0, S, b_pos_ }, { 1, S, b_hinge_v_ }, { 2, S, b_coef_ }, { 3, S, b_blocks_ }, { 4, U, u_block_ } });
	set_psd_ = dev_.uniform_set(sh_psd_, { { 0, S, b_blocks_ }, { 1, U, u_psd_ } });
	set_gather_ = dev_.uniform_set(sh_gather_, { { 0, S, b_ptr_ }, { 1, S, b_src_ }, { 2, S, b_blocks_ }, { 3, S, b_weight_ }, { 4, S, b_values_ }, { 5, U, u_gather_ } });
	if (!set_block_.index || !set_psd_.index || !set_gather_.index)
		return bail("uniform sets");
	psd_last_ = 1;
	sign_last_ = 1.0f;
	open_ = true;
	return true;
}

void GpuHessian::close() {
	if (!dev_.ok())
		return;
	if (!open_) {
		dev_.close();
		return;
	}
	for (::RID *r : { &set_block_, &set_psd_, &set_gather_ })
		if (r->index) {
			dev_.free_rid(*r);
			*r = ::RID();
		}
	for (::RID *r : { &b_pos_, &b_hinge_v_, &b_coef_, &b_blocks_, &b_ptr_, &b_src_, &b_weight_, &b_values_, &u_block_, &u_psd_, &u_gather_,
				 &pipe_block_, &pipe_psd_, &pipe_gather_, &sh_block_, &sh_psd_, &sh_gather_ })
		if (r->index) {
			dev_.free_rid(*r);
			*r = ::RID();
		}
	dev_.close();
	open_ = false;
}

bool GpuHessian::run(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, bool with_gather, std::string *err) {
	if (!open_)
		return false;
	if (x.size() != p.n || uint32_t(p.n_hinges) != n_hinges_) {
		if (err)
			*err = "problem changed under the device";
		return false;
	}
	const int64_t t0 = rdc::host_usec();
	p.positions(x, pos_);
	if (!dev_.buffer_update(b_pos_, 0, pos_.size() * sizeof(Df), pos_.data())) {
		if (err)
			*err = dev_.error();
		return false;
	}
	if (sign != sign_last_) {
		const BlockParams bp{ n_hinges_, sign, { 0, 0 } };
		if (!dev_.buffer_update(u_block_, 0, sizeof bp, &bp)) {
			if (err)
				*err = dev_.error();
			return false;
		}
		sign_last_ = sign;
	}
	if (uint32_t(psd) != psd_last_) {
		const PsdParams pp{ n_hinges_, psd ? 1u : 0u, { 0, 0 } };
		if (!dev_.buffer_update(u_psd_, 0, sizeof pp, &pp)) {
			if (err)
				*err = dev_.error();
			return false;
		}
		psd_last_ = uint32_t(psd);
	}
	const int64_t t1 = rdc::host_usec();
	if (timestamps_)
		dev_.capture_timestamp("fit_gpu_begin");
	if (!dev_.list_begin()) {
		if (err)
			*err = dev_.error();
		return false;
	}
	dev_.bind_pipeline(pipe_block_);
	dev_.bind_uniform_set(set_block_);
	dev_.dispatch(rdc::Device::groups_for(n_hinges_, 64));
	dev_.barrier();
	dev_.bind_pipeline(pipe_psd_);
	dev_.bind_uniform_set(set_psd_);
	dev_.dispatch(rdc::Device::groups_for(n_hinges_, 64));
	if (with_gather) {
		dev_.barrier();
		dev_.bind_pipeline(pipe_gather_);
		dev_.bind_uniform_set(set_gather_);
		dev_.dispatch(rdc::Device::groups_for(nnz_, 64));
	}
	dev_.list_end();
	if (timestamps_)
		dev_.capture_timestamp("fit_gpu_end");
	dev_.submit();
	dev_.sync();
	const int64_t t2 = rdc::host_usec();
	if (timestamps_) {
		const int64_t n = dev_.timestamps_count();
		if (n >= 2)
			stats_.gpu_ns += dev_.timestamp_gpu_ns(n - 1) - dev_.timestamp_gpu_ns(n - 2);
	}
	stats_.upload_us += t1 - t0;
	stats_.gpu_us += t2 - t1;
	return true;
}

bool GpuHessian::assemble(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, polyfem::StiffnessMatrix &out) {
	const int64_t t0 = rdc::host_usec();
	std::string err;
	if (!run(p, x, psd, sign, true, &err)) {
		stats_.fallbacks++;
		stats_.last_error = err.empty() ? dev_.error() : err;
		return false;
	}
	const int64_t t2 = rdc::host_usec();
	if (!dev_.buffer_get_into(b_values_, 0, values_.size() * sizeof(Df), values_.data())) {
		stats_.fallbacks++;
		stats_.last_error = dev_.error();
		return false;
	}
	p.assemble(values_, out);
	const int64_t t3 = rdc::host_usec();
	stats_.readback_us += t3 - t2;
	stats_.last_us = t3 - t0;
	stats_.total_us += stats_.last_us;
	stats_.assemblies++;
	return true;
}

bool GpuHessian::blocks(const SimProblem &p, const Eigen::VectorXd &x, bool psd, float sign, std::vector<Df> &blocks) {
	std::string err;
	if (!run(p, x, psd, sign, false, &err)) {
		stats_.last_error = err.empty() ? dev_.error() : err;
		return false;
	}
	blocks.assign(size_t(p.n_hinges) * kBlock, Df{});
	if (!dev_.buffer_get_into(b_blocks_, 0, blocks.size() * sizeof(Df), blocks.data())) {
		stats_.last_error = dev_.error();
		return false;
	}
	return true;
}

} // namespace fit_gpu
