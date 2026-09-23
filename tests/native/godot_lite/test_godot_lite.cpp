// Native test of godot-lite (guest/godot_lite + vendor/godot-core-subset),
// built with the host clang++ by build.sh. Compiled the way Cassie TUs are:
// Godot include paths, gdl_prelude.h force-included, no namespace written.
//
//   test_godot_lite.exe <curve3d_ref_godot-4.7.2.txt>
//
// Exit 0 when every check passes. Each section also runs a negative control
// that must fail, so a check that cannot fail is caught.

#include "core/math/dynamic_bvh.h"
#include "core/math/geometry_3d.h"
#include "core/object/ref_counted.h"
#include "core/object/worker_thread_pool.h"
#include "core/string/print_string.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/templates/rb_map.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "scene/resources/curve.h"
#include "scene/resources/mesh.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(m_cond)                                                          \
	do {                                                                       \
		g_checks++;                                                            \
		if (!(m_cond)) {                                                       \
			g_failed++;                                                        \
			fprintf(stdout, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #m_cond); \
		}                                                                      \
	} while (0)

// A negative control: the expression must be false.
#define CONTROL(m_cond) CHECK(!(m_cond))

static void section(const char *p_name) {
	fprintf(stdout, "-- %s\n", p_name);
}

// ---------------------------------------------------------------------------

static void test_vector_cow() {
	section("Vector copy-on-write");
	Vector<int> a = { 1, 2, 3 };
	Vector<int> b = a;
	CHECK(b.shares_storage_with(a));
	CHECK(a.ptr() == b.ptr()); // one buffer

	// Reads never detach.
	const Vector<int> &cb = b;
	CHECK(cb[0] == 1 && cb.size() == 3 && cb.find(3) == 2 && cb.has(2));
	CHECK(b.shares_storage_with(a));

	// The first write detaches the writer only.
	b.write[0] = 9;
	CHECK(!b.shares_storage_with(a));
	CHECK(a[0] == 1 && b[0] == 9);
	CONTROL(a[0] == 9); // control: a shared (non-COW) buffer would show 9 here

	Vector<int> c = a;
	c.ptrw()[1] = 7; // ptrw detaches too
	CHECK(a[1] == 2 && c[1] == 7);
	Vector<int> d = a;
	d.push_back(4);
	d.remove_at(0);
	CHECK(a.size() == 3 && d.size() == 3 && d[0] == 2 && d[2] == 4);
	Vector<int> e = a;
	e.resize(5);
	CHECK(a.size() == 3 && e.size() == 5 && e[3] == 0 && e[4] == 0);

	// Unshared writes do not copy.
	Vector<int> f = { 5, 4, 3 };
	const int *before = f.ptr();
	f.write[0] = 6;
	CHECK(f.ptr() == before);

	// append_array with itself (aliasing source).
	Vector<int> g = { 1, 2 };
	g.append_array(g);
	CHECK(g.size() == 4 && g[2] == 1 && g[3] == 2);

	// sort / sort_custom / insert / erase / slice / reverse.
	Vector<int> h = { 3, 1, 2 };
	Vector<int> h0 = h;
	h.sort();
	CHECK(h[0] == 1 && h[1] == 2 && h[2] == 3);
	CHECK(h0[0] == 3); // sorting detached
	struct Greater {
		bool operator()(int p_a, int p_b) const { return p_a > p_b; }
	};
	h.sort_custom<Greater>();
	CHECK(h[0] == 3 && h[2] == 1);
	h.insert(1, 10);
	CHECK(h.size() == 4 && h[1] == 10);
	CHECK(h.erase(10) && h.size() == 3);
	Vector<int> s = h.slice(1);
	CHECK(s.size() == 2 && s[0] == 2);
	h.reverse();
	CHECK(h[0] == 1);

	// Vector<bool> keeps real pointers.
	Vector<bool> vb;
	vb.resize(3);
	vb.write[1] = true;
	CHECK(!vb[0] && vb[1] && vb.ptr()[1]);

	// A Packed array inside a Variant is a value: the copy out is COW.
	PackedVector3Array pv = { Vector3(1, 2, 3) };
	Variant var = pv;
	PackedVector3Array out = var;
	CHECK(out.shares_storage_with(pv));
	out.write[0] = Vector3();
	CHECK(PackedVector3Array(var)[0] == Vector3(1, 2, 3));
}

// ---------------------------------------------------------------------------

static void test_hash_map_order() {
	section("HashMap / HashSet insertion order");
	HashMap<int, String> m;
	const int keys[] = { 5, 3, 9, 1, 7 };
	for (int k : keys) {
		m.insert(k, itos(k));
	}
	std::vector<int> order;
	for (const KeyValue<int, String> &kv : m) {
		order.push_back(kv.key);
	}
	CHECK(order == std::vector<int>({ 5, 3, 9, 1, 7 }));
	CONTROL(order == std::vector<int>({ 1, 3, 5, 7, 9 })); // control: not sorted order

	m.erase(3);
	m[5] = "five"; // updating in place keeps the position
	m.insert(3, "3");
	order.clear();
	for (const KeyValue<int, String> &kv : m) {
		order.push_back(kv.key);
	}
	CHECK(order == std::vector<int>({ 5, 9, 1, 7, 3 }));
	CHECK(m[5] == "five" && m.has(3) && !m.has(4));

	// Many keys through several rehashes: iteration == insertion order.
	HashMap<int64_t, int> big;
	std::vector<int64_t> inserted;
	std::mt19937_64 rng(12345);
	for (int i = 0; i < 5000; i++) {
		const int64_t k = int64_t(rng() % 1000003);
		if (!big.has(k)) {
			inserted.push_back(k);
		}
		big[k] = i;
	}
	std::vector<int64_t> seen;
	for (const KeyValue<int64_t, int> &kv : big) {
		seen.push_back(kv.key);
	}
	CHECK(seen == inserted);
	CHECK(big.size() == inserted.size());

	// String keys (Cassie's cycle signatures) and HashSet.
	HashMap<String, int> sm;
	sm.insert("b,c", 1);
	sm.insert("a,b", 2);
	CHECK(sm.has(String("b,") + "c") && sm["a,b"] == 2);
	HashSet<String> hs;
	hs.insert("x");
	hs.insert("y");
	hs.insert("x");
	CHECK(hs.size() == 2 && hs.has("y") && !hs.has("z"));

	// Iterator find, as CassieSketcher uses it.
	HashMap<int, int> empty_map;
	HashMap<int, int>::Iterator none = empty_map.find(1);
	CHECK(!none);

	// RBMap sorts, LocalVector sorts.
	RBMap<real_t, int> rb;
	rb.insert(2.0f, 2);
	rb.insert(0.5f, 1);
	rb.insert(3.0f, 3);
	CHECK(rb.front()->key() == 0.5f && rb.back()->key() == 3.0f && rb.size() == 3);
	LocalVector<int> lv;
	lv.push_back(3);
	lv.push_back(1);
	lv.push_back(2);
	lv.sort();
	CHECK(lv[0] == 1 && lv[2] == 3);
}

// ---------------------------------------------------------------------------

static void test_array_dictionary_aliasing() {
	section("Array / Dictionary shared references");
	Array a;
	Array b = a;
	b.push_back(1);
	CHECK(a.size() == 1 && int(a[0]) == 1);
	Array c = a.duplicate();
	c.push_back(2);
	CHECK(a.size() == 1 && c.size() == 2);
	CONTROL(a.size() == 2); // control: duplicate() must not alias

	// Through a Variant the Array is still the same one.
	Variant va = a;
	Array a2 = va;
	a2.push_back("x");
	CHECK(a.size() == 2 && String(a[1]) == "x");

	// TypedArray is an Array.
	TypedArray<RefCounted> ta;
	Array untyped = ta;
	untyped.push_back(Variant());
	CHECK(ta.size() == 1);

	Dictionary d;
	Dictionary e = d;
	e["k"] = 1;
	CHECK(d.has("k") && int(d["k"]) == 1);
	Dictionary f = d.duplicate();
	f["k"] = 2;
	CHECK(int(d["k"]) == 1 && int(f["k"]) == 2);
	CONTROL(int(d["k"]) == 2); // control: duplicate() must not alias

	// Nested containers are shared, a deep duplicate is not.
	Array inner;
	d["inner"] = inner;
	Array inner_back = d["inner"];
	inner_back.push_back(42);
	CHECK(inner.size() == 1);
	Dictionary deep = d.duplicate(true);
	Array deep_inner = deep["inner"];
	deep_inner.push_back(43);
	CHECK(inner.size() == 1 && deep_inner.size() == 2);

	// Keys keep insertion order; int and float keys are distinct; get() defaults.
	Dictionary o;
	o["z"] = 1;
	o["a"] = 2;
	o[3] = 3;
	o[3.0] = 4;
	Array keys = o.keys();
	CHECK(keys.size() == 4 && String(keys[0]) == "z" && String(keys[1]) == "a");
	CHECK(int(o[3]) == 3 && int(o[3.0]) == 4);
	CHECK(int(o.get("missing", -1)) == -1);
	CHECK(bool(o.get("z", false)));
	int visited = 0;
	for (const Variant *k = o.next(nullptr); k != nullptr; k = o.next(k)) {
		visited++;
	}
	CHECK(visited == 4);
	CHECK(o.erase("a") && o.size() == 3 && !o.has("a"));

	// Variant conversions Cassie relies on.
	Dictionary r;
	r["on_surface"] = true;
	r["projected"] = Vector3(1, 2, 3);
	r["distance"] = real_t(0.5);
	r["patch_id"] = 7;
	CHECK(bool(r.get("on_surface", false)));
	const Vector3 p = r.get("projected", Vector3());
	CHECK(p == Vector3(1, 2, 3));
	CHECK(real_t(r["distance"]) == 0.5f && int(r.get("patch_id", -1)) == 7);
	CHECK(Variant(1) == Variant(1.0) && !Variant(1).hash_compare(Variant(1.0)));
	CHECK(Variant(Array()).get_type() == Variant::ARRAY && Variant(r).get_type() == Variant::DICTIONARY);
	PackedInt32Array from_array = Variant(Array({ 1, 2, 3 }));
	CHECK(from_array.size() == 3 && from_array[2] == 3);
	CHECK(vformat("n=%d x=%.2f s=%s", 3, 1.5, "ok") == "n=3 x=1.50 s=ok");
}

// ---------------------------------------------------------------------------

static int g_alive = 0;

class Probe : public RefCounted {
	GDCLASS(Probe, RefCounted);

public:
	int value = 0;
	Vector3 scale(const Vector3 &p_v, real_t p_s) const { return p_v * p_s * real_t(value); }
	Dictionary project(const Vector3 &p_pos) const {
		Dictionary d;
		d["on_surface"] = p_pos.y > 0;
		d["projected"] = Vector3(p_pos.x, 0, p_pos.z);
		return d;
	}
	Probe() { g_alive++; }
	~Probe() override { g_alive--; }
};

static void test_objects() {
	section("Ref / Variant(Object) / Callable / Mesh / WorkerThreadPool");
	{
		Ref<Probe> r;
		r.instantiate();
		CHECK(g_alive == 1 && r->get_reference_count() == 1);
		{
			Variant v = r; // a Variant holds a reference
			CHECK(r->get_reference_count() == 2);
			Ref<Probe> back = v;
			CHECK(back == r && r->get_reference_count() == 3);
			Ref<RefCounted> base = back; // upcast
			Ref<Probe> down = base; // downcast
			CHECK(down.is_valid() && down->get_class() == "Probe");
		}
		CHECK(r->get_reference_count() == 1);
		Array arr;
		arr.push_back(r);
		r.unref();
		CHECK(g_alive == 1); // the Array keeps it
		arr.clear();
		CHECK(g_alive == 0);
	}
	CONTROL(g_alive != 0);

	Ref<Probe> p;
	p.instantiate();
	p->value = 2;
	Callable c = callable_mp(p.ptr(), &Probe::scale);
	CHECK(c.is_valid());
	CHECK(Vector3(c.call(Vector3(1, 2, 3), real_t(0.5))) == Vector3(1, 2, 3));
	// callp with the CallError path, as OnSurfaceEnergy uses it.
	Callable proj = callable_mp(p.ptr(), &Probe::project);
	Variant res;
	Callable::CallError err;
	const Variant arg = Vector3(1, 2, 3);
	const Variant *args[1] = { &arg };
	proj.callp(args, 1, res, err);
	CHECK(err.error == Callable::CallError::CALL_OK && res.get_type() == Variant::DICTIONARY);
	CHECK(Vector3(Dictionary(res).get("projected", Vector3())) == Vector3(1, 0, 3));
	proj.callp(args, 0, res, err);
	CHECK(err.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS);
	// By-name callables need ClassDB, which godot-lite does not have.
	Callable by_name(p.ptr(), "project");
	CoreGlobals::print_error_enabled = false;
	by_name.callp(args, 1, res, err);
	CHECK(err.error == Callable::CallError::CALL_ERROR_INVALID_METHOD);
	CHECK(!Callable().is_valid());

	// ArrayMesh round trip (CassieSurfaceManager -> CassieSurfacePatch).
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = PackedVector3Array({ Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0) });
	arrays[Mesh::ARRAY_INDEX] = PackedInt32Array({ 0, 1, 2 });
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	Ref<Mesh> as_mesh = mesh;
	CHECK(as_mesh->get_surface_count() == 1);
	const Array got = as_mesh->surface_get_arrays(0);
	CHECK(PackedVector3Array(got[Mesh::ARRAY_VERTEX]).size() == 3 && PackedInt32Array(got[Mesh::ARRAY_INDEX])[2] == 2);
	CHECK(got[Mesh::ARRAY_NORMAL].get_type() == Variant::NIL);

	// The synchronous pool has run the task by the time add_native_task returns.
	int ran = 0;
	WorkerThreadPool *wtp = WorkerThreadPool::get_singleton();
	WorkerThreadPool::TaskID tid = wtp->add_native_task([](void *p_ud) { (*(int *)p_ud)++; }, &ran, false, "t");
	CHECK(ran == 1 && wtp->is_task_completed(tid) && wtp->wait_for_task_completion(tid) == OK);

	// DynamicBVH + Geometry3D, as CassieSurfacePatch / intersection_finder use them.
	DynamicBVH bvh;
	DynamicBVH::ID ids[3];
	for (int i = 0; i < 3; i++) {
		ids[i] = bvh.insert(AABB(Vector3(real_t(i * 10), 0, 0), Vector3(1, 1, 1)), (void *)(intptr_t)(i + 1));
	}
	struct Hits {
		int count = 0;
		intptr_t last = 0;
		bool operator()(void *p_data) {
			count++;
			last = (intptr_t)p_data;
			return false;
		}
	} hits;
	bvh.aabb_query(AABB(Vector3(9.5, 0, 0), Vector3(1, 1, 1)), hits);
	CHECK(hits.count == 1 && hits.last == 2);
	bvh.remove(ids[1]);
	hits = Hits();
	bvh.aabb_query(AABB(Vector3(9.5, 0, 0), Vector3(1, 1, 1)), hits);
	CHECK(hits.count == 0);
	Vector3 ps, qt;
	Geometry3D::get_closest_points_between_segments(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(1, 1, -1), Vector3(1, 1, 1), ps, qt);
	CHECK(ps.is_equal_approx(Vector3(1, 0, 0)) && qt.is_equal_approx(Vector3(1, 1, 0)));
}

// ---------------------------------------------------------------------------
// Curve3D: Godot's own unit-test expectations (tests/scene/test_curve_3d.cpp
// in entities-godot), an independent arc-length integral, and a line-by-line
// comparison with the same probes run in Godot 4.7.2 (curve3d_ref.gd).

static Vector3 bezier(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_d, double p_t) {
	const double s = 1.0 - p_t;
	return p_a * real_t(s * s * s) + p_b * real_t(3 * s * s * p_t) + p_c * real_t(3 * s * p_t * p_t) + p_d * real_t(p_t * p_t * p_t);
}

static std::string v3s(const Vector3 &p_v) {
	char buf[96];
	snprintf(buf, sizeof(buf), "%.6f %.6f %.6f", p_v.x, p_v.y, p_v.z);
	return buf;
}

static void dump(std::vector<std::string> &r_lines, const char *p_name, const Ref<Curve3D> &p_c) {
	char buf[256];
	const real_t L = p_c->get_baked_length();
	snprintf(buf, sizeof(buf), "%s length %.6f", p_name, L);
	r_lines.push_back(buf);
	snprintf(buf, sizeof(buf), "%s baked_points %d", p_name, int(p_c->get_baked_points().size()));
	r_lines.push_back(buf);
	for (float f : { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f }) {
		snprintf(buf, sizeof(buf), "%s sample_baked %.2f lin %s", p_name, f, v3s(p_c->sample_baked(L * f, false)).c_str());
		r_lines.push_back(buf);
		snprintf(buf, sizeof(buf), "%s sample_baked %.2f cub %s", p_name, f, v3s(p_c->sample_baked(L * f, true)).c_str());
		r_lines.push_back(buf);
		snprintf(buf, sizeof(buf), "%s up %.2f %s", p_name, f, v3s(p_c->sample_baked_up_vector(L * f)).c_str());
		r_lines.push_back(buf);
	}
	snprintf(buf, sizeof(buf), "%s closest_offset %.6f", p_name, p_c->get_closest_offset(Vector3(30, 20, 10)));
	r_lines.push_back(buf);
	snprintf(buf, sizeof(buf), "%s closest_point %s", p_name, v3s(p_c->get_closest_point(Vector3(30, 20, 10))).c_str());
	r_lines.push_back(buf);
	snprintf(buf, sizeof(buf), "%s tessellate %d", p_name, int(p_c->tessellate().size()));
	r_lines.push_back(buf);
	snprintf(buf, sizeof(buf), "%s tessellate_even %d", p_name, int(p_c->tessellate_even_length().size()));
	r_lines.push_back(buf);
}

// Same words, numbers within p_tol. Returns the largest numeric difference.
static bool lines_match(const std::string &p_a, const std::string &p_b, double p_tol, double &r_maxdiff) {
	std::istringstream sa(p_a), sb(p_b);
	std::string ta, tb;
	while (true) {
		const bool ga = bool(sa >> ta);
		const bool gb = bool(sb >> tb);
		if (ga != gb) {
			return false;
		}
		if (!ga) {
			return true;
		}
		char *ea = nullptr;
		char *eb = nullptr;
		const double na = strtod(ta.c_str(), &ea);
		const double nb = strtod(tb.c_str(), &eb);
		if (*ea == 0 && *eb == 0 && ea != ta.c_str()) {
			const double d = std::fabs(na - nb);
			r_maxdiff = std::max(r_maxdiff, d);
			if (d > p_tol) {
				return false;
			}
		} else if (ta != tb) {
			return false;
		}
	}
}

static Ref<Curve3D> make_bezier() {
	Ref<Curve3D> a = memnew(Curve3D);
	a->add_point(Vector3(0, 0, 0), Vector3(), Vector3(50, 0, 0));
	a->add_point(Vector3(0, 50, 0), Vector3(-50, 0, -50), Vector3());
	return a;
}

static Ref<Curve3D> make_closed() {
	Ref<Curve3D> b = memnew(Curve3D);
	b->set_bake_interval(0.05);
	b->add_point(Vector3(0, 0, 0), Vector3(), Vector3(0.5, 0, 0));
	b->add_point(Vector3(1, 1, 0), Vector3(0, -0.5, 0), Vector3(0, 0.5, 0.2));
	b->add_point(Vector3(0, 2, 0.5), Vector3(0.4, 0, 0), Vector3());
	b->set_closed(true);
	return b;
}

static void test_curve3d(const char *p_ref_path) {
	section("Curve3D baking");
	{
		// Godot's own expectations (test_curve_3d.cpp).
		Ref<Curve3D> c = memnew(Curve3D);
		c->add_point(Vector3());
		CHECK(c->get_baked_length() == 0 && c->get_baked_points().size() == 1);
		c->add_point(Vector3(0, 50, 0));
		CHECK(Math::is_equal_approx(c->get_baked_length(), 50));
		CHECK(c->get_baked_points().size() == 369);
		CHECK(c->get_baked_tilts().size() == 369 && c->get_baked_up_vectors().size() == 369);
		CHECK(c->sample(0, 0.5) == Vector3(0, 25, 0) && c->samplef(1) == Vector3(0, 50, 0));
		CHECK(c->sample_baked(c->get_closest_offset(Vector3(0, 25, 0))) == Vector3(0, 25, 0));
		CHECK(c->sample_baked(c->get_closest_offset(Vector3(0, 25, 0)), true) == Vector3(0, 25, 0));
		CHECK(c->sample_baked_with_rotation(c->get_closest_offset(Vector3(0, 25, 0))) == Transform3D(Basis(Vector3(0, 0, -1), Vector3(1, 0, 0), Vector3(0, -1, 0)), Vector3(0, 25, 0)));
		CHECK(c->sample_baked_up_vector(c->get_closest_offset(Vector3(0, 25, 0))) == Vector3(1, 0, 0));
		CHECK(c->get_closest_point(Vector3(50, 25, 0)) == Vector3(0, 25, 0));
		CHECK(c->get_closest_point(Vector3(0, 100, 0)) == Vector3(0, 50, 0));
		CONTROL(c->get_closest_point(Vector3(50, 25, 0)) == Vector3(50, 25, 0)); // control

		Ref<Curve3D> off = memnew(Curve3D); // regression #81879
		off->add_point(Vector3());
		off->add_point(Vector3(0, .1, 1));
		CHECK((off->sample_baked_up_vector(off->get_closest_offset(Vector3(0, 0, .9))) - Vector3(0, 0.995037, -0.099504)).length() < 0.01);

		Ref<Curve3D> cross = memnew(Curve3D); // regression #88923
		cross->add_point(Vector3(), Vector3(-1, 0, 2), Vector3(1, 0, 0));
		cross->add_point(Vector3(1, 0, 0), Vector3(-1, 0, 0), Vector3(1, 0, 2));
		CHECK(cross->get_baked_points().size() >= 3);
		CHECK(cross->sample_baked_with_rotation(cross->get_closest_offset(Vector3(0.5, 0, 0))).is_equal_approx(Transform3D(Basis(Vector3(0, 0, 1), Vector3(0, 1, 0), Vector3(-1, 0, 0)), Vector3(0.5, 0, 0))));

		Ref<Curve3D> t = make_bezier();
		const int def = t->tessellate().size();
		CHECK(t->tessellate(6).size() > def && t->tessellate(4).size() < def);
		CHECK(t->tessellate(5, 5).size() < def && t->tessellate(5, 3).size() > def);
		const int even = t->tessellate_even_length().size();
		t->add_point(Vector3(0, 150, 0));
		CHECK(t->tessellate_even_length().size() > even + 5);

		const real_t len = make_bezier()->get_baked_length();
		Ref<Curve3D> coarse = make_bezier();
		coarse->set_bake_interval(10.0);
		CHECK(coarse->get_baked_length() < len);
		coarse->set_up_vector_enabled(false);
		CHECK(coarse->get_baked_up_vectors().size() == 0);
	}
	{
		// Independent: the arc length of the same cubic by a 200k-segment
		// polyline (double precision), and sample_baked(L/2) on the curve.
		const Vector3 p0(0, 0, 0), c0(50, 0, 0), c1(-50, 0, -50), p3(0, 50, 0);
		const int N = 200000;
		double arc = 0.0;
		Vector3 prev = p0;
		std::vector<Vector3> dense;
		dense.reserve(N + 1);
		dense.push_back(p0);
		for (int i = 1; i <= N; i++) {
			const Vector3 q = bezier(p0, p0 + c0, p3 + c1, p3, double(i) / N);
			arc += (q - prev).length();
			prev = q;
			dense.push_back(q);
		}
		Ref<Curve3D> b = make_bezier();
		const double L = b->get_baked_length();
		fprintf(stdout, "   bezier baked length %.6f, integral %.6f, rel %.2e\n", L, arc, std::fabs(L - arc) / arc);
		CHECK(std::fabs(L - arc) / arc < 1e-3);
		const Vector3 mid = b->sample_baked(real_t(L * 0.5));
		double best = 1e30;
		for (const Vector3 &q : dense) {
			best = std::min(best, double((q - mid).length()));
		}
		CHECK(best < 0.05);
		CONTROL(std::fabs(L - 50.0) / 50.0 < 1e-3); // control: not the straight-line length
	}
	{
		// Godot 4.7.2 ran the same probes (curve3d_ref.gd); compare every line.
		std::vector<std::string> ours;
		dump(ours, "bezier", make_bezier());
		dump(ours, "closed", make_closed());
		std::ifstream in(p_ref_path ? p_ref_path : "");
		std::vector<std::string> ref;
		for (std::string ln; std::getline(in, ln);) {
			if (!ln.empty() && ln.back() == '\r') {
				ln.pop_back();
			}
			if (!ln.empty()) {
				ref.push_back(ln);
			}
		}
		CHECK(ref.size() == ours.size() && !ref.empty());
		double maxdiff = 0.0;
		int bad = 0;
		for (size_t i = 0; i < ref.size() && i < ours.size(); i++) {
			if (!lines_match(ours[i], ref[i], 1e-4, maxdiff)) {
				bad++;
				fprintf(stdout, "   mismatch: ours [%s] godot [%s]\n", ours[i].c_str(), ref[i].c_str());
			}
		}
		fprintf(stdout, "   %d/%d lines match Godot 4.7.2 (max |diff| %.2e)\n", int(ref.size()) - bad, int(ref.size()), maxdiff);
		CHECK(bad == 0);
		// Control: a perturbed curve must not match the reference.
		std::vector<std::string> moved;
		Ref<Curve3D> m = make_bezier();
		m->set_point_position(1, Vector3(0, 50.5, 0));
		dump(moved, "bezier", m);
		int moved_bad = 0;
		for (size_t i = 0; i < moved.size() && i < ref.size(); i++) {
			double ignored = 0.0;
			moved_bad += lines_match(moved[i], ref[i], 1e-4, ignored) ? 0 : 1;
		}
		CONTROL(moved_bad == 0);
	}
}

int main(int argc, char **argv) {
	test_vector_cow();
	test_hash_map_order();
	test_array_dictionary_aliasing();
	test_objects();
	test_curve3d(argc > 1 ? argv[1] : nullptr);
	fprintf(stdout, "%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
	return g_failed ? 1 : 0;
}
