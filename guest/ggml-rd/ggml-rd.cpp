// ggml-rd: the registry, device, buffer type, buffer and backend interfaces
// (ggml backend API v2, vendor/ggml/src/ggml-backend-impl.h). The graph
// itself is rd_graph.cpp; pipelines and uniform sets are rd_kernels.cpp.
#include "rd_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_rd {

Ctx &ctx() {
	static Ctx c;
	return c;
}

bool device_ok() {
	const Ctx &c = ctx();
	return c.dev != nullptr && c.dev->ok();
}

void set_error(const std::string &e) {
	ctx().last_error = e;
	GGML_LOG_ERROR("ggml-rd: %s\n", e.c_str());
}

void ensure_idle() {
	Ctx &c = ctx();
	if (!c.pending) {
		return;
	}
	++c.st.waits;
	if (c.hooks.wait_gpu) {
		c.hooks.wait_gpu(c.hooks.user); // returns on a later frame
	}
	c.dev->sync();
	c.pending = false;
	if (c.ts_armed) {
		// A local device resolves a submit's timestamps at its sync: the two
		// graph_compute captured are the last two.
		const int64_t n = c.dev->timestamps_count();
		c.last_gpu_ns = n >= 2 ? c.dev->timestamp_gpu_ns(n - 1) - c.dev->timestamp_gpu_ns(n - 2) : -1;
		c.ts_armed = false;
	}
}

void coop() {
	Ctx &c = ctx();
	++c.st.coop_yields;
	if (c.hooks.coop) {
		c.hooks.coop(c.hooks.user);
	}
}

// --- buffer -----------------------------------------------------------------

static const ggml_backend_buffer_i &buffer_iface();

Buffer *buffer_of(const ggml_tensor *t) {
	if (t == nullptr) {
		return nullptr;
	}
	ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
	if (b == nullptr || b->buft != buft()) {
		return nullptr;
	}
	return static_cast<Buffer *>(b->context);
}

uint64_t byte_offset(const ggml_tensor *t, const Buffer *b) {
	return uint64_t(uintptr_t(t->data) - b->base);
}

static Buffer *buf(ggml_backend_buffer_t b) {
	return static_cast<Buffer *>(b->context);
}

static void buffer_free(ggml_backend_buffer_t buffer) {
	Buffer *b = buf(buffer);
	Ctx &c = ctx();
	if (device_ok() && b->rid.index != 0) {
		ensure_idle();
		forget_sets_of(b->rid);
		c.dev->free_rid(b->rid);
	}
	--c.st.buffers_live;
	c.st.bytes_live -= int64_t(b->size);
	delete b;
}

static void *buffer_get_base(ggml_backend_buffer_t buffer) {
	return reinterpret_cast<void *>(buf(buffer)->base);
}

static void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor, const void *data, size_t offset, size_t size) {
	Buffer *b = buf(buffer);
	ensure_idle();
	Ctx &c = ctx();
	if (!c.dev->buffer_update(b->rid, byte_offset(tensor, b) + offset, size, data)) {
		set_error("set_tensor " + std::string(tensor->name) + ": " + c.dev->error());
	}
	c.st.set_bytes += int64_t(size);
}

static void buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
	Buffer *b = buf(buffer);
	ensure_idle();
	Ctx &c = ctx();
	if (!c.dev->buffer_get_into(b->rid, byte_offset(tensor, b) + offset, size, data)) {
		set_error("get_tensor " + std::string(tensor->name) + ": " + c.dev->error());
		std::memset(data, 0, size);
	}
	c.st.get_bytes += int64_t(size);
}

// A non-zero or unaligned fill goes up as bytes, 16 MiB at a time.
static void fill_bytes(Buffer *b, uint64_t offset, uint64_t size, uint8_t value) {
	Ctx &c = ctx();
	const uint64_t chunk = uint64_t(16) << 20;
	std::vector<uint8_t> v(size_t(std::min(size, chunk)), value);
	for (uint64_t done = 0; done < size;) {
		const uint64_t n = std::min(size - done, chunk);
		if (!c.dev->buffer_update(b->rid, offset + done, n, v.data())) {
			set_error("fill: " + c.dev->error());
			return;
		}
		done += n;
	}
}

static void buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor, uint8_t value, size_t offset, size_t size) {
	Buffer *b = buf(buffer);
	ensure_idle();
	const uint64_t off = byte_offset(tensor, b) + offset;
	if (value == 0 && off % 4 == 0 && size % 4 == 0) {
		ctx().dev->buffer_clear(b->rid, off, size);
		++ctx().st.clears;
	} else {
		fill_bytes(b, off, size, value);
	}
}

static bool buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *src, ggml_tensor *dst) {
	Buffer *sb = buffer_of(src);
	if (sb == nullptr) {
		return false; // ggml falls back to get + set
	}
	Buffer *db = buf(buffer);
	ensure_idle();
	Ctx &c = ctx();
	++c.st.copies;
	return c.dev->buffer_copy(sb->rid, db->rid, ggml_nbytes(src), byte_offset(src, sb), byte_offset(dst, db));
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
	Buffer *b = buf(buffer);
	ensure_idle();
	if (value == 0) {
		ctx().dev->buffer_clear(b->rid, 0, b->size);
		++ctx().st.clears;
	} else {
		fill_bytes(b, 0, b->size, value);
	}
}

static const ggml_backend_buffer_i &buffer_iface() {
	static const ggml_backend_buffer_i i = [] {
		ggml_backend_buffer_i x{};
		x.free_buffer = buffer_free;
		x.get_base = buffer_get_base;
		x.init_tensor = nullptr;
		x.memset_tensor = buffer_memset_tensor;
		x.set_tensor = buffer_set_tensor;
		x.get_tensor = buffer_get_tensor;
		x.set_tensor_2d = nullptr;
		x.get_tensor_2d = nullptr;
		x.cpy_tensor = buffer_cpy_tensor;
		x.clear = buffer_clear;
		x.reset = nullptr;
		return x;
	}();
	return i;
}

// --- buffer type ------------------------------------------------------------

static size_t max_buffer_bytes() {
	// 2 GiB by default (portable maxStorageBufferRange); Gate 0F showed
	// 4 GiB - 256 working on the RTX 4090, so it can be raised.
	const char *e = std::getenv("GGML_RD_MAX_BUFFER_MB");
	const long long mb = e ? std::atoll(e) : 0;
	if (mb > 0) {
		return std::min<size_t>(size_t(mb) << 20, size_t(0xFFFFFF00u));
	}
	return size_t(2048) << 20;
}

static const char *buft_get_name(ggml_backend_buffer_type_t) {
	return GGML_RD_NAME;
}

static ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t bt, size_t size) {
	if (!device_ok()) {
		set_error("alloc_buffer: no RenderingDevice");
		return nullptr;
	}
	ensure_idle();
	Ctx &c = ctx();
	const size_t bytes = (std::max<size_t>(size, 1) + 255) & ~size_t(255);
	if (bytes > max_buffer_bytes()) {
		set_error("alloc_buffer: " + std::to_string(bytes) + " bytes is over the maximum");
		return nullptr;
	}
	::RID rid = c.dev->storage_buffer_empty(bytes, true);
	if (rid.index == 0) {
		set_error("alloc_buffer: " + c.dev->error());
		return nullptr;
	}
	Buffer *b = new Buffer;
	b->rid = rid;
	b->size = bytes;
	b->index = c.next_buffer_index++;
	b->base = uintptr_t(0x1000) + (uintptr_t(b->index) << 40);
	++c.st.buffers_live;
	c.st.bytes_live += int64_t(bytes);
	return ggml_backend_buffer_init(bt, buffer_iface(), b, size);
}

static size_t buft_get_alignment(ggml_backend_buffer_type_t) {
	return 256;
}

static size_t buft_get_max_size(ggml_backend_buffer_type_t) {
	return max_buffer_bytes();
}

// 16 bytes of slack per tensor: a kernel that loads 32-bit words (f16 and
// bf16 are read two to a word) never reads past its tensor's allocation.
static size_t buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor *t) {
	return (ggml_nbytes(t) + 15) & ~size_t(15);
}

static bool buft_is_host(ggml_backend_buffer_type_t) {
	return false;
}

// --- device -----------------------------------------------------------------

static ggml_backend_device &device_obj();
static ggml_backend_reg &reg_obj();

ggml_backend_buffer_type_t buft() {
	static ggml_backend_buffer_type t = [] {
		ggml_backend_buffer_type x{};
		x.iface.get_name = buft_get_name;
		x.iface.alloc_buffer = buft_alloc_buffer;
		x.iface.get_alignment = buft_get_alignment;
		x.iface.get_max_size = buft_get_max_size;
		x.iface.get_alloc_size = buft_get_alloc_size;
		x.iface.is_host = buft_is_host;
		x.device = &device_obj();
		x.context = nullptr;
		return x;
	}();
	return &t;
}

static const char *dev_get_name(ggml_backend_dev_t) {
	return GGML_RD_NAME "0";
}

static const char *dev_get_description(ggml_backend_dev_t) {
	static std::string desc;
	if (desc.empty() && device_ok()) {
		desc = "Godot RenderingDevice (" + ctx().dev->device_name() + ")";
	}
	return desc.empty() ? "Godot RenderingDevice" : desc.c_str();
}

static void dev_get_memory(ggml_backend_dev_t, size_t *free, size_t *total) {
	const Ctx &c = ctx();
	*total = c.total_bytes;
	const size_t used = size_t(std::max<int64_t>(c.st.bytes_live, 0));
	*free = c.total_bytes > used ? c.total_bytes - used : 0;
}

static enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) {
	return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void dev_get_props(ggml_backend_dev_t d, ggml_backend_dev_props *props) {
	std::memset(props, 0, sizeof(*props));
	props->name = dev_get_name(d);
	props->description = dev_get_description(d);
	props->type = dev_get_type(d);
	props->device_id = nullptr;
	dev_get_memory(d, &props->memory_free, &props->memory_total);
	props->caps.async = false;
	props->caps.host_buffer = false;
	props->caps.buffer_from_host_ptr = false;
	props->caps.events = false;
}

static ggml_backend_t dev_init_backend(ggml_backend_dev_t d, const char *params);

static ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t) {
	return buft();
}

static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor *op) {
	switch (op->op) {
		case GGML_OP_NONE:
		case GGML_OP_RESHAPE:
		case GGML_OP_VIEW:
		case GGML_OP_PERMUTE:
		case GGML_OP_TRANSPOSE:
			return true;
		default:
			return find_op(op) != nullptr;
	}
}

static bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t bt) {
	return bt == buft();
}

static bool dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
	return false;
}

static ggml_backend_device &device_obj() {
	static ggml_backend_device d = [] {
		ggml_backend_device x{};
		x.iface.get_name = dev_get_name;
		x.iface.get_description = dev_get_description;
		x.iface.get_memory = dev_get_memory;
		x.iface.get_type = dev_get_type;
		x.iface.get_props = dev_get_props;
		x.iface.init_backend = dev_init_backend;
		x.iface.get_buffer_type = dev_get_buffer_type;
		x.iface.get_host_buffer_type = nullptr;
		x.iface.buffer_from_host_ptr = nullptr;
		x.iface.supports_op = dev_supports_op;
		x.iface.supports_buft = dev_supports_buft;
		x.iface.offload_op = dev_offload_op;
		x.iface.event_new = nullptr;
		x.iface.event_free = nullptr;
		x.iface.event_synchronize = nullptr;
		x.reg = &reg_obj();
		x.context = nullptr;
		return x;
	}();
	return d;
}

// --- backend ----------------------------------------------------------------

static ggml_guid *backend_guid() {
	static ggml_guid g = { 0x52, 0x44, 0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x72, 0x64, 0x2d, 0x67, 0x6f, 0x64, 0x6f, 0x74, 0x01 };
	return &g;
}

static const char *backend_get_name(ggml_backend_t) {
	return GGML_RD_NAME "0";
}

static void backend_free(ggml_backend_t backend) {
	if (device_ok()) {
		ensure_idle();
	}
	delete backend;
}

static void backend_synchronize(ggml_backend_t) {
	if (device_ok()) {
		ensure_idle();
	}
}

static enum ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph *cgraph) {
	return graph_compute(cgraph);
}

static ggml_backend_t dev_init_backend(ggml_backend_dev_t d, const char *) {
	if (!device_ok()) {
		return nullptr;
	}
	ggml_backend_i i{};
	i.get_name = backend_get_name;
	i.free = backend_free;
	i.set_tensor_async = nullptr;
	i.get_tensor_async = nullptr;
	i.set_tensor_2d_async = nullptr;
	i.get_tensor_2d_async = nullptr;
	i.cpy_tensor_async = nullptr;
	i.synchronize = backend_synchronize;
	i.graph_plan_create = nullptr;
	i.graph_plan_free = nullptr;
	i.graph_plan_update = nullptr;
	i.graph_plan_compute = nullptr;
	i.graph_compute = backend_graph_compute;
	i.event_record = nullptr;
	i.event_wait = nullptr;
	i.graph_optimize = nullptr;
	return new ggml_backend{ backend_guid(), i, d, nullptr };
}

// --- registry ---------------------------------------------------------------

static const char *reg_get_name(ggml_backend_reg_t) {
	return GGML_RD_NAME;
}

static size_t reg_get_device_count(ggml_backend_reg_t) {
	return device_ok() ? 1 : 0;
}

static ggml_backend_dev_t reg_get_device(ggml_backend_reg_t, size_t index) {
	GGML_ASSERT(index == 0 && device_ok());
	return &device_obj();
}

static ggml_backend_reg &reg_obj() {
	static ggml_backend_reg r = [] {
		ggml_backend_reg x{};
		x.api_version = GGML_BACKEND_API_VERSION;
		x.iface.get_name = reg_get_name;
		x.iface.get_device_count = reg_get_device_count;
		x.iface.get_device = reg_get_device;
		x.iface.get_proc_address = nullptr;
		x.context = nullptr;
		return x;
	}();
	return r;
}

} // namespace ggml_rd

using namespace ggml_rd;

ggml_backend_reg_t ggml_backend_rd_reg(void) {
	return &reg_obj();
}

void ggml_backend_rd_attach(rdc::Device *dev, size_t total_bytes) {
	Ctx &c = ctx();
	c.dev = dev;
	c.total_bytes = total_bytes;
	c.pending = false;
}

void ggml_backend_rd_set_hooks(const ggml_rd_hooks &hooks) {
	ctx().hooks = hooks;
}

bool ggml_backend_is_rd(ggml_backend_t backend) {
	return backend != nullptr && ggml_guid_matches(backend->guid, backend_guid());
}

bool ggml_backend_buffer_is_rd(ggml_backend_buffer_t buffer) {
	return buffer != nullptr && buffer->buft == buft();
}

::RID ggml_backend_rd_buffer_rid(ggml_backend_buffer_t buffer) {
	return ggml_backend_buffer_is_rd(buffer) ? static_cast<Buffer *>(buffer->context)->rid : ::RID();
}

size_t ggml_backend_rd_tensor_offset(const ggml_tensor *tensor) {
	Buffer *b = buffer_of(tensor);
	return b ? size_t(byte_offset(tensor, b)) : 0;
}

void ggml_backend_rd_ensure_idle(void) {
	if (device_ok()) {
		ensure_idle();
	}
}

bool ggml_backend_rd_tensor_upload(const ggml_tensor *tensor, size_t offset, const std::string &path,
		uint64_t file_offset, size_t bytes) {
	Ctx &c = ctx();
	Buffer *b = buffer_of(tensor);
	if (!device_ok() || b == nullptr) {
		set_error(std::string("upload ") + tensor->name + ": not in an RD buffer");
		return false;
	}
	if (offset + bytes > ggml_nbytes(tensor)) {
		set_error(std::string("upload ") + tensor->name + ": past the end of the tensor");
		return false;
	}
	if (c.hooks.upload == nullptr) {
		set_error("upload: no upload hook");
		return false;
	}
	ensure_idle(); // a local device drops what is recorded between submit and sync
	if (!c.hooks.upload(c.hooks.user, path, file_offset, bytes, b->rid, byte_offset(tensor, b) + offset)) {
		set_error("upload " + path + ": the host did not serve it");
		return false;
	}
	c.st.set_bytes += int64_t(bytes);
	return true;
}

std::string ggml_backend_rd_stats(void) {
	const Stats &s = ctx().st;
	char b[768];
	std::snprintf(b, sizeof b,
			"graphs=%lld nodes=%lld dispatches=%lld skipped=%lld barriers=%lld failed_graphs=%lld "
			"last{nodes=%lld dispatches=%lld barriers=%lld skipped=%lld} "
			"pipelines=%lld set0_created=%lld slots_created=%lld coop_yields=%lld waits=%lld faults=%lld "
			"params_bytes=%lld set_bytes=%lld get_bytes=%lld copies=%lld clears=%lld buffers_live=%lld bytes_live=%lld",
			(long long)s.graphs, (long long)s.nodes, (long long)s.dispatches, (long long)s.skipped,
			(long long)s.barriers, (long long)s.failed_graphs, (long long)s.last_nodes,
			(long long)s.last_dispatches, (long long)s.last_barriers, (long long)s.last_skipped,
			(long long)s.pipelines, (long long)s.set0_created, (long long)s.slots_created,
			(long long)s.coop_yields, (long long)s.waits, (long long)s.faults, (long long)s.params_bytes,
			(long long)s.set_bytes, (long long)s.get_bytes, (long long)s.copies, (long long)s.clears,
			(long long)s.buffers_live, (long long)s.bytes_live);
	return b;
}

void ggml_backend_rd_last_graph(int64_t *dispatches, int64_t *barriers) {
	*dispatches = ctx().st.last_dispatches;
	*barriers = ctx().st.last_barriers;
}

void ggml_backend_rd_set_timestamps(bool on) {
	ctx().timestamps = on;
}

int64_t ggml_backend_rd_last_gpu_ns(void) {
	return ctx().last_gpu_ns;
}

std::string ggml_backend_rd_last_error(void) {
	return ctx().last_error;
}

void ggml_backend_rd_set_profile(int level) {
	ctx().profile_level = level;
}

const ggml_rd_profile &ggml_backend_rd_last_profile(void) {
	return ctx().profile;
}

const char *ggml_backend_rd_kernel_name(int kernel) {
	return kernel >= 0 && kernel < int(sizeof kKernels / sizeof kKernels[0]) ? kKernels[kernel].name : "?";
}

std::string ggml_backend_rd_last_dropped(void) {
	return ctx().last_dropped;
}

void ggml_backend_rd_release(void) {
	if (device_ok()) {
		ensure_idle();
	}
	release_kernels();
}
