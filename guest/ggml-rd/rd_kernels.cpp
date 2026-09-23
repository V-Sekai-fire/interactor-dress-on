// ggml-rd: kernels, pipelines, the params table and the uniform-set caches.
//
// Every kernel binds the same six descriptors (lean/Ggml/SlangCodegen/
// Common.lean, checked by kernels/ggml/gen_ggml_kernel_table.py), so a
// uniform set built against one kernel's shader binds under every pipeline:
//  - set 0, per (params, s0, s1, s2, dst) buffer tuple, cached by their RIDs
//    until one of them is freed (Godot frees a set with any buffer it binds);
//  - set 1, per params slot: a 16-byte uniform buffer holding base = 64 * i,
//    created on first use and kept, so the k-th dispatch of every graph binds
//    the same slot set.
// Creating a set is a handful of host calls; after every 256 the COOP hook
// gives the frame back.
#include "rd_internal.h"

#include <array>
#include <cstring>
#include <map>

#include "ggml_kernels.inc" // the embedded SPIR-V (kernels/ggml/gen.sh, build dir)

namespace ggml_rd {

namespace {

struct Kern {
	::RID shader, pipeline;
};

struct State {
	std::vector<Kern> kerns;
	// Any kernel's shader: uniform sets are created against it (and die with
	// it, so it is freed last).
	::RID layout_shader;
	::RID params;
	uint32_t params_slots = 0;
	std::vector<::RID> slot_ubos, slot_sets;
	std::map<std::array<int64_t, 5>, ::RID> set0;
	uint32_t created_since_coop = 0;
};

State &S() {
	static State s;
	return s;
}

void created_one() {
	State &s = S();
	if (++s.created_since_coop >= 256) {
		s.created_since_coop = 0;
		coop();
	}
}

} // namespace

::RID kernel_pipeline(int k) {
	State &s = S();
	if (k < 0 || uint32_t(k) >= kKernelCount) {
		set_error("kernel id " + std::to_string(k) + " out of range");
		return ::RID();
	}
	if (s.kerns.size() < kKernelCount) {
		s.kerns.resize(kKernelCount);
	}
	Kern &kn = s.kerns[k];
	if (kn.pipeline.index != 0) {
		return kn.pipeline;
	}
	const ggml_kernels::Entry *e = ggml_kernels::find(kKernels[k].name);
	if (e == nullptr) {
		set_error(std::string("kernel ") + kKernels[k].name + " is not embedded");
		return ::RID();
	}
	rdc::Device &d = *ctx().dev;
	kn.shader = d.shader_from_spirv(e->bytes, e->size);
	if (kn.shader.index == 0) {
		set_error(std::string("shader ") + kKernels[k].name + ": " + d.error());
		return ::RID();
	}
	kn.pipeline = d.compute_pipeline(kn.shader);
	if (kn.pipeline.index == 0) {
		set_error(std::string("pipeline ") + kKernels[k].name + ": " + d.error());
		d.free_rid(kn.shader);
		kn.shader = ::RID();
		return ::RID();
	}
	if (s.layout_shader.index == 0) {
		s.layout_shader = kn.shader;
	}
	++ctx().st.pipelines;
	return kn.pipeline;
}

::RID params_buffer(uint32_t slots) {
	State &s = S();
	if (s.params.index != 0 && s.params_slots >= slots) {
		return s.params;
	}
	uint32_t cap = 8192;
	while (cap < slots) {
		cap *= 2;
	}
	rdc::Device &d = *ctx().dev;
	if (s.params.index != 0) {
		// Every set-0 set binds the table; Godot frees them all with it.
		for (auto &kv : s.set0) {
			d.forget(kv.second);
		}
		s.set0.clear();
		d.free_rid(s.params);
		s.params = ::RID();
		s.params_slots = 0;
	}
	s.params = d.storage_buffer_empty(size_t(cap) * kSlotBytes, true);
	if (s.params.index == 0) {
		set_error("params table: " + d.error());
		return ::RID();
	}
	s.params_slots = cap;
	return s.params;
}

::RID slot_set(uint32_t i) {
	State &s = S();
	rdc::Device &d = *ctx().dev;
	if (s.layout_shader.index == 0) {
		set_error("slot_set before any pipeline");
		return ::RID();
	}
	while (s.slot_sets.size() <= i) {
		const uint32_t j = uint32_t(s.slot_sets.size());
		const uint32_t data[4] = { j * kWordsPerSlot, 0, 0, 0 };
		::RID ubo = d.uniform_buffer(sizeof data, data);
		if (ubo.index == 0) {
			set_error("slot ubo: " + d.error());
			return ::RID();
		}
		::RID set = d.uniform_set(s.layout_shader, { { B_SLOT, rdc::UNIFORM_TYPE_UNIFORM_BUFFER, ubo } }, SET_SLOT);
		if (set.index == 0) {
			set_error("slot set: " + d.error());
			d.free_rid(ubo);
			return ::RID();
		}
		s.slot_ubos.push_back(ubo);
		s.slot_sets.push_back(set);
		++ctx().st.slots_created;
		created_one();
	}
	return s.slot_sets[i];
}

::RID tensor_set(::RID params, ::RID s0, ::RID s1, ::RID s2, ::RID dst) {
	State &s = S();
	const std::array<int64_t, 5> key = { params.index, s0.index, s1.index, s2.index, dst.index };
	auto it = s.set0.find(key);
	if (it != s.set0.end()) {
		return it->second;
	}
	rdc::Device &d = *ctx().dev;
	if (s.layout_shader.index == 0) {
		set_error("tensor_set before any pipeline");
		return ::RID();
	}
	const int SB = rdc::UNIFORM_TYPE_STORAGE_BUFFER;
	::RID set = d.uniform_set(s.layout_shader,
			{ { B_PARAMS, SB, params }, { B_SRC0, SB, s0 }, { B_SRC1, SB, s1 }, { B_SRC2, SB, s2 }, { B_DST, SB, dst } },
			SET_TENSORS);
	if (set.index == 0) {
		set_error("set 0: " + d.error());
		return ::RID();
	}
	s.set0.emplace(key, set);
	++ctx().st.set0_created;
	created_one();
	return set;
}

void forget_sets_of(::RID buffer) {
	State &s = S();
	rdc::Device &d = *ctx().dev;
	for (auto it = s.set0.begin(); it != s.set0.end();) {
		const auto &k = it->first;
		if (k[1] == buffer.index || k[2] == buffer.index || k[3] == buffer.index || k[4] == buffer.index) {
			d.forget(it->second);
			it = s.set0.erase(it);
		} else {
			++it;
		}
	}
}

void release_kernels() {
	State &s = S();
	if (ctx().dev == nullptr) {
		s = State{};
		return;
	}
	rdc::Device &d = *ctx().dev;
	const bool live = d.ok();
	auto drop = [&](::RID &r) {
		if (r.index != 0) {
			if (live) {
				d.free_rid(r);
			} else {
				d.forget(r);
			}
			r = ::RID();
		}
	};
	// Sets first (they depend on the shader and the buffers), shaders last.
	for (auto &kv : s.set0) {
		drop(kv.second);
	}
	for (::RID &r : s.slot_sets) {
		drop(r);
	}
	for (::RID &r : s.slot_ubos) {
		drop(r);
	}
	drop(s.params);
	for (Kern &k : s.kerns) {
		drop(k.pipeline);
		drop(k.shader);
	}
	s = State{};
}

} // namespace ggml_rd
