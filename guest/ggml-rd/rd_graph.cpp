// ggml-rd: graph_compute. One graph is one compute list and one submit:
//
//   1. wait for the previous graph if it is still in flight (the wait hook;
//      AGENTS.md rule 4 means that is a later frame than its submit);
//   2. pack every node into its params slot (64 words, 256 B), skipping the
//      layout-only ops, and note the byte ranges it reads and writes;
//   3. make sure every pipeline, slot set and set-0 set exists (cached;
//      COOP every 256 created);
//   4. one buffer_update of the whole params table, before list_begin
//      (Godot refuses buffer_update inside a compute list);
//   5. record: per dispatch bind pipeline (if it changed), set 0 (if it
//      changed), the slot's set 1, dispatch; a barrier only when the
//      dispatch reads bytes written, or writes bytes read or written, by a
//      dispatch since the last barrier (Godot's render graph does not order
//      dispatches inside one list, AGENTS.md facts; GGML_RD_BARRIER_ALL=1
//      puts one after every dispatch);
//   6. submit, mark pending, return. Nothing waits here.
//
// Controls and measurement (Gate 3): GGML_RD_FAULT moves a source of every
// n-th dispatch; GGML_RD_DROP_BARRIER=<k> records the k-th barrier elision
// places as if it were there (the segments reset) but leaves it out of the
// list; GGML_RD_PROFILE=<1|2> reads the host clock around the phases (and,
// at 2, around each dispatch's packing and recording).
#include "rd_internal.h"

#include <cstdlib>
#include <cstring>

namespace ggml_rd {

namespace {

struct Range {
	int64_t rid;
	uint64_t lo, hi; // [lo, hi) bytes
};

struct Dispatch {
	int kernel;
	uint32_t groups[3];
	::RID bind[4]; // s0, s1, s2, dst buffers
	::RID set0;
	Range reads[3];
	int nreads;
	Range write;
	const ggml_tensor *node;
};

bool overlaps(const std::vector<Range> &v, const Range &r) {
	for (const Range &x : v) {
		if (x.rid == r.rid && x.lo < r.hi && r.lo < x.hi) {
			return true;
		}
	}
	return false;
}

int env_int(const char *name) {
	const char *e = std::getenv(name);
	return e ? std::atoi(e) : 0;
}

// Byte offset of t in its RD buffer (the OffsetFn fill_standard asks).
bool rd_offset(const ggml_tensor *t, void *, uint64_t *off) {
	const Buffer *b = buffer_of(t);
	if (b == nullptr) {
		return false;
	}
	*off = byte_offset(t, b);
	return true;
}

// Widened to whole 4-byte words: kernels address uint words, and a 16-bit
// destination's read-modify-write (lean/Ggml/SlangCodegen/Move.lean) stores
// the neighbouring half of its first and last word too.
Range range_of(const ggml_tensor *t, const Buffer *b) {
	const uint64_t lo = byte_offset(t, b);
	return Range{ b->rid.index, lo & ~uint64_t(3), (lo + ggml_nbytes(t) + 3) & ~uint64_t(3) };
}

std::string node_desc(const ggml_tensor *n) {
	std::string s = ggml_op_desc(n);
	s += " '";
	s += n->name;
	s += "' ";
	s += ggml_type_name(n->type);
	return s;
}

ggml_status fail_graph(const std::string &why) {
	++ctx().st.failed_graphs;
	set_error(why);
	return GGML_STATUS_FAILED;
}

} // namespace

ggml_status graph_compute(ggml_cgraph *g) {
	Ctx &c = ctx();
	if (!device_ok()) {
		return fail_graph("graph_compute: no RenderingDevice");
	}
	ensure_idle();
	rdc::Device &d = *c.dev;

	const bool barrier_all = env_int("GGML_RD_BARRIER_ALL") != 0;
	const int fault = env_int("GGML_RD_FAULT");
	const int drop_barrier = env_int("GGML_RD_DROP_BARRIER");
	const char *pe = std::getenv("GGML_RD_PROFILE");
	const int prof = pe != nullptr ? std::atoi(pe) : c.profile_level;
	ggml_rd_profile &pf = c.profile;
	pf = ggml_rd_profile{};
	pf.level = prof;
	c.last_dropped.clear();
	auto clock = [prof]() -> int64_t { return prof > 0 ? rdc::host_usec() : 0; };
	const int64_t t_entry = clock();

	static std::vector<uint32_t> table;
	static std::vector<Dispatch> ds;
	ds.clear();
	int64_t skipped = 0;

	for (int i = 0; i < g->n_nodes; ++i) {
		ggml_tensor *node = g->nodes[i];
		if (is_layout_only(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
			++skipped;
			continue;
		}
		const int64_t t_node0 = prof > 1 ? rdc::host_usec() : 0;
		const OpEntry *e = find_op(node);
		if (e == nullptr) {
			return fail_graph("no packer accepts " + node_desc(node));
		}
		const size_t slot = ds.size();
		if (table.size() < (slot + 1) * kWordsPerSlot) {
			table.resize((slot + 1) * kWordsPerSlot * 2);
		}
		uint32_t *w = table.data() + slot * kWordsPerSlot;
		std::string why;
		if (!fill_standard(w, node, rd_offset, nullptr, &why)) {
			return fail_graph(why + ": " + node_desc(node));
		}

		Dispatch dp{};
		dp.node = node;
		Buffer *db = buffer_of(node);
		dp.write = range_of(node, db);
		dp.bind[3] = db->rid;
		for (int s = 0; s < 3; ++s) {
			const ggml_tensor *src = node->src[s];
			if (src == nullptr) {
				dp.bind[s] = db->rid; // an unused binding names the dst buffer: no new dependency
				continue;
			}
			Buffer *sb = buffer_of(src);
			dp.bind[s] = sb->rid;
			dp.reads[dp.nreads++] = range_of(src, sb);
		}

		Pack p;
		p.node = node;
		p.w = w;
		if (!e->pack(p) || p.kernel < 0 || p.groups[0] == 0 || p.groups[1] == 0 || p.groups[2] == 0) {
			return fail_graph(std::string("packer ") + e->name + " refused " + node_desc(node));
		}
		if (prof > 1) {
			ggml_rd_profile_dispatch pd;
			pd.node = node;
			pd.kernel = p.kernel;
			pd.pack_us = t_node0;
			pf.per_dispatch.push_back(pd);
		}
		w[W_KERNEL] = uint32_t(p.kernel);
		dp.kernel = p.kernel;
		std::memcpy(dp.groups, p.groups, sizeof dp.groups);

		++c.dispatch_serial;
		if (fault > 0 && c.dispatch_serial % uint64_t(fault) == 0) {
			// The control: read one element past where the kernel should.
			w[(node->src[1] ? W_SRC1 : W_SRC0) + T_OFF] += 1;
			++c.st.faults;
		}
		ds.push_back(dp);
		if (prof > 1) {
			pf.per_dispatch.back().pack_us = rdc::host_usec() - pf.per_dispatch.back().pack_us;
		}
	}
	const int64_t t_packed = clock();

	c.st.graphs++;
	c.st.nodes += g->n_nodes;
	c.st.skipped += skipped;
	c.st.last_nodes = g->n_nodes;
	c.st.last_skipped = skipped;
	c.st.last_dispatches = int64_t(ds.size());
	c.st.last_barriers = 0;
	pf.nodes = g->n_nodes;
	pf.skipped = skipped;
	pf.dispatches = int64_t(ds.size());
	pf.us_pack = t_packed - t_entry;
	if (ds.empty()) {
		pf.us_total = clock() - t_entry;
		return GGML_STATUS_SUCCESS;
	}

	// Everything the recording binds exists before the list opens.
	for (Dispatch &dp : ds) {
		if (kernel_pipeline(dp.kernel).index == 0) {
			return fail_graph("pipeline for " + node_desc(dp.node) + " failed");
		}
	}
	const uint32_t n = uint32_t(ds.size());
	::RID params = params_buffer(n);
	if (params.index == 0) {
		return fail_graph("params table: " + c.last_error);
	}
	for (uint32_t i = 0; i < n; ++i) {
		if (slot_set(i).index == 0) {
			return fail_graph("slot set " + std::to_string(i) + ": " + c.last_error);
		}
	}
	for (Dispatch &dp : ds) {
		dp.set0 = tensor_set(params, dp.bind[0], dp.bind[1], dp.bind[2], dp.bind[3]);
		if (dp.set0.index == 0) {
			return fail_graph("set 0 for " + node_desc(dp.node) + ": " + c.last_error);
		}
	}
	const int64_t t_prepared = clock();
	// A COOP yield above ended a frame, but nothing was submitted, so the
	// device is still idle here.
	if (!d.buffer_update(params, 0, size_t(n) * kSlotBytes, table.data())) {
		return fail_graph("params upload: " + d.error());
	}
	c.st.params_bytes += int64_t(n) * kSlotBytes;
	const int64_t t_uploaded = clock();

	std::vector<Range> seg_reads, seg_writes;
	seg_reads.reserve(16);
	seg_writes.reserve(16);
	int64_t barriers = 0;
	int64_t placed = 0; // barriers elision placed, dropped one included
	int bound_kernel = -1;
	int64_t bound_set0 = 0;
	const bool ts = c.timestamps || env_int("GGML_RD_TIMESTAMPS") != 0;
	if (ts) {
		d.capture_timestamp("ggml_rd_graph_begin");
	}
	d.list_begin();
	for (uint32_t i = 0; i < n; ++i) {
		const int64_t t_rec0 = prof > 1 ? rdc::host_usec() : 0;
		const Dispatch &dp = ds[i];
		bool hazard = barrier_all && i > 0;
		for (int r = 0; r < dp.nreads && !hazard; ++r) {
			hazard = overlaps(seg_writes, dp.reads[r]); // RAW
		}
		if (!hazard) {
			hazard = overlaps(seg_writes, dp.write) || overlaps(seg_reads, dp.write); // WAW, WAR
		}
		if (hazard) {
			++placed;
			if (drop_barrier > 0 && placed == drop_barrier && !barrier_all) {
				// The control: this barrier is required (a hazard), and is left out.
				c.last_dropped = "barrier " + std::to_string(placed) + " before dispatch " + std::to_string(i) +
						" " + node_desc(dp.node);
			} else {
				d.barrier(); // Godot re-binds the pipeline and sets after it
				++barriers;
			}
			seg_reads.clear();
			seg_writes.clear();
			if (prof > 1) {
				pf.per_dispatch[i].barrier = true;
			}
		}
		for (int r = 0; r < dp.nreads; ++r) {
			seg_reads.push_back(dp.reads[r]);
		}
		seg_writes.push_back(dp.write);
		if (dp.kernel != bound_kernel) {
			d.bind_pipeline(kernel_pipeline(dp.kernel));
			bound_kernel = dp.kernel;
		}
		if (dp.set0.index != bound_set0) {
			d.bind_uniform_set(dp.set0, SET_TENSORS);
			bound_set0 = dp.set0.index;
		}
		d.bind_uniform_set(slot_set(i), SET_SLOT);
		d.dispatch(dp.groups[0], dp.groups[1], dp.groups[2]);
		if (prof > 1) {
			pf.per_dispatch[i].record_us = rdc::host_usec() - t_rec0;
		}
	}
	d.list_end();
	const int64_t t_recorded = clock();
	if (ts) {
		d.capture_timestamp("ggml_rd_graph_end");
		c.ts_armed = true;
	}
	d.submit();
	c.pending = true;
	const int64_t t_submitted = clock();
	pf.barriers = barriers;
	pf.us_prepare = t_prepared - t_packed;
	pf.us_upload = t_uploaded - t_prepared;
	pf.us_record = t_recorded - t_uploaded;
	pf.us_submit = t_submitted - t_recorded;
	pf.us_total = t_submitted - t_entry;

	c.st.dispatches += n;
	c.st.barriers += barriers;
	c.st.last_barriers = barriers;
	return GGML_STATUS_SUCCESS;
}

} // namespace ggml_rd
