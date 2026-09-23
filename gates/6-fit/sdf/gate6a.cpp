// Gate 6a: the brick-grid SDF and the spline sampler against OpenVDB.
//
//   gate6a <target_avatar.obj> <faces.obj> <openvdb_dump.jsonl> [--voxel h] [--label name]
//
// The dump is tools/fit/openvdb_dump run on the same avatar (vertices from the
// oracle run's target_avatar.obj, faces from the input avatar.obj) at the
// points of gates/6-fit/sdf/make_points.py: per point the world p, OpenVDB's
// index coordinate, floor, uvw, its 4^3 stencil and sampleHessian's x, g, h.
//
// (a) sampler: the reference (spline_ref.h) on OpenVDB's own stencils and
//     uvw reproduces sampleHessian to <= 1 ULP in all 13 outputs; our
//     world->index (p * (1/h), floor, subtract) reproduces OpenVDB's bitwise.
//     When the Lean emit is compiled in (FIT_KERNELS_PENDING unset), the kernel
//     is held to the same bound against the reference.
// (b) grid: our SdfGrid sampled at each point vs OpenVDB, over the samples
//     with |v_vdb| < 5 voxels: |dv| <= 0.25 voxel for >= 99%, sign agreement
//     >= 99.5%, gradient angle <= 10 deg for >= 95%.
// (c) control: the same comparison with our grid shifted +0.5 voxel in x
//     (sampled at p - 0.5h e_x) must FAIL (b).
// Also: brick count, bytes, distance queries, fill time, and a fill-order
// check (a fresh grid filled walking the points in reverse gives bitwise the
// same samples).
#include <polyfem/solver/forms/garment_forms/SdfGrid.hpp>
#include <polyfem/solver/forms/garment_forms/SdfSpline.hpp>

#include "spline_ref.h"

#include <igl/default_num_threads.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using polyfem::solver::SdfGrid;
namespace sdf_spline = polyfem::solver::sdf_spline;

namespace
{
	struct Rec
	{
		double p[3], ijk_f[3], uvw[3];
		int base[3];
		double stencil[64];
		double x, g[3], h[9];
	};

	bool read_obj(const std::string &path, std::vector<double> &v, std::vector<int> &f)
	{
		std::ifstream in(path);
		if (!in)
			return false;
		std::string line;
		while (std::getline(in, line))
		{
			if (line.size() > 2 && line[0] == 'v' && line[1] == ' ')
			{
				const char *s = line.c_str() + 2;
				char *e = nullptr;
				for (int d = 0; d < 3; d++)
				{
					v.push_back(std::strtod(s, &e));
					s = e;
				}
			}
			else if (line.size() > 2 && line[0] == 'f' && line[1] == ' ')
			{
				std::istringstream ss(line.substr(2));
				std::string tok;
				std::vector<int> poly;
				while (ss >> tok)
				{
					const int idx = std::atoi(tok.c_str());
					poly.push_back(idx > 0 ? idx - 1 : int(v.size() / 3) + idx);
				}
				for (size_t k = 1; k + 1 < poly.size(); k++)
				{
					f.push_back(poly[0]);
					f.push_back(poly[k]);
					f.push_back(poly[k + 1]);
				}
			}
		}
		return true;
	}

	// Distance in units in the last place, via the sign-magnitude -> ordered map.
	std::uint64_t ulps(const double a, const double b)
	{
		auto ord = [](double d) {
			std::int64_t i;
			std::memcpy(&i, &d, 8);
			return i < 0 ? std::int64_t(0x8000000000000000ull - std::uint64_t(i)) : i;
		};
		const std::int64_t ia = ord(a), ib = ord(b);
		return ia > ib ? std::uint64_t(ia) - std::uint64_t(ib) : std::uint64_t(ib) - std::uint64_t(ia);
	}

	bool same_bits(const double a, const double b) { return std::memcmp(&a, &b, 8) == 0; }

	double quantile(std::vector<double> v, const double q)
	{
		if (v.empty())
			return NAN;
		std::sort(v.begin(), v.end());
		const size_t k = std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5));
		return v[k];
	}

	int n_pass = 0, n_fail = 0;

	void out(const char *fmt, ...)
		__attribute__((format(printf, 1, 2)));
	void out(const char *fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		std::vprintf(fmt, ap);
		va_end(ap);
		std::fflush(stdout);
	}

	void verdict(const bool ok, const std::string &name, const std::string &detail)
	{
		(ok ? n_pass : n_fail)++;
		out("%s %s: %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
	}

	std::string fmt(const char *f, ...)
		__attribute__((format(printf, 1, 2)));
	std::string fmt(const char *f, ...)
	{
		char buf[1024];
		va_list ap;
		va_start(ap, f);
		std::vsnprintf(buf, sizeof buf, f, ap);
		va_end(ap);
		return buf;
	}

	struct Compare
	{
		size_t n = 0, near = 0, within = 0, sign_near = 0, sign_all = 0, ang_n = 0, ang_ok = 0, ang_skipped = 0;
		std::vector<double> dv_near, ang;
		double frac_within() const { return near ? double(within) / near : NAN; }
		double frac_sign() const { return near ? double(sign_near) / near : NAN; }
		double frac_ang() const { return ang_n ? double(ang_ok) / ang_n : NAN; }
		bool pass() const { return frac_within() >= 0.99 && frac_sign() >= 0.995 && frac_ang() >= 0.95; }
	};

	// One sample of our grid at world point q through the reference sampler.
	void sample_ours(const SdfGrid &grid, const double q[3], double o[10])
	{
		double st[64], uvw[3];
		grid.sample_point(q, st, uvw);
		gate6a::spline_hessian_ref(st, uvw, o);
	}

	Compare compare(const std::vector<Rec> &recs, const std::vector<std::array<double, 10>> &ours, const double h)
	{
		Compare c;
		c.n = recs.size();
		for (size_t n = 0; n < recs.size(); n++)
		{
			const Rec &r = recs[n];
			const auto &o = ours[n];
			const bool sgn = (o[0] > 0) == (r.x > 0);
			c.sign_all += sgn;
			if (!(std::abs(r.x) < 5 * h))
				continue;
			c.near++;
			c.sign_near += sgn;
			const double dv = std::abs(o[0] - r.x) / h;
			c.dv_near.push_back(dv);
			c.within += dv <= 0.25;
			const double na = std::sqrt(o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
			const double nb = std::sqrt(r.g[0] * r.g[0] + r.g[1] * r.g[1] + r.g[2] * r.g[2]);
			if (!(na > 1e-12) || !(nb > 1e-12))
			{
				c.ang_skipped++;
				continue;
			}
			const double cs = std::clamp((o[1] * r.g[0] + o[2] * r.g[1] + o[3] * r.g[2]) / (na * nb), -1.0, 1.0);
			const double deg = std::acos(cs) * (180.0 / 3.14159265358979323846);
			c.ang.push_back(deg);
			c.ang_n++;
			c.ang_ok += deg <= 10.0;
		}
		return c;
	}

	void report(const char *tag, const Compare &c)
	{
		out("%s: samples %zu, |v_vdb|<5 voxels %zu; |dv|<=0.25 voxel %.4f%% (p50 %.4g p99 %.4g max %.4g voxel); "
			"sign %.4f%% (all samples %.4f%%); angle<=10deg %.4f%% of %zu (p50 %.3g p95 %.3g max %.3g deg, %zu zero-gradient skipped)\n",
			tag, c.n, c.near, 100 * c.frac_within(), quantile(c.dv_near, 0.5), quantile(c.dv_near, 0.99),
			quantile(c.dv_near, 1.0), 100 * c.frac_sign(), 100.0 * c.sign_all / std::max<size_t>(c.n, 1),
			100 * c.frac_ang(), c.ang_n, quantile(c.ang, 0.5), quantile(c.ang, 0.95), quantile(c.ang, 1.0), c.ang_skipped);
	}
} // namespace

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		std::fprintf(stderr, "usage: gate6a <target_avatar.obj> <faces.obj> <dump.jsonl> [--voxel h] [--label name]\n");
		return 2;
	}
	double h = 0.01;
	std::string label = "points";
	for (int a = 4; a + 1 < argc; a += 2)
	{
		if (!std::strcmp(argv[a], "--voxel"))
			h = std::strtod(argv[a + 1], nullptr);
		else if (!std::strcmp(argv[a], "--label"))
			label = argv[a + 1];
	}
	igl::default_num_threads(1);

	std::vector<double> av, fv;
	std::vector<int> af, ff;
	if (!read_obj(argv[1], av, af) || !read_obj(argv[2], fv, ff) || ff.empty() || fv.size() != av.size())
	{
		std::fprintf(stderr, "gate6a: cannot read avatar/faces\n");
		return 2;
	}
	Eigen::MatrixXd V(av.size() / 3, 3);
	for (Eigen::Index i = 0; i < V.rows(); i++)
		V.row(i) << av[3 * i], av[3 * i + 1], av[3 * i + 2];
	Eigen::MatrixXi F(ff.size() / 3, 3);
	for (Eigen::Index i = 0; i < F.rows(); i++)
		F.row(i) << ff[3 * i], ff[3 * i + 1], ff[3 * i + 2];

	// --- dump ---------------------------------------------------------------
	std::vector<Rec> recs;
	{
		std::ifstream in(argv[3]);
		if (!in)
		{
			std::fprintf(stderr, "gate6a: cannot open %s\n", argv[3]);
			return 2;
		}
		std::string line;
		std::getline(in, line);
		const auto hdr = nlohmann::json::parse(line);
		if (hdr["voxel_size"].get<double>() != h || hdr["n_verts"].get<long>() != V.rows() || hdr["n_tris"].get<long>() != F.rows())
		{
			std::fprintf(stderr, "gate6a: dump header does not match the avatar/voxel\n");
			return 2;
		}
		out("[%s] OpenVDB grid: %ld active voxels, %ld leaves, %ld bytes, built in %.2f s (openvdb_dump, TBB)\n",
			label.c_str(), hdr["active_voxels"].get<long>(), hdr["leaf_nodes"].get<long>(), hdr["mem_bytes"].get<long>(),
			hdr["build_seconds"].get<double>());
		size_t errors = 0;
		while (std::getline(in, line))
		{
			const auto j = nlohmann::json::parse(line);
			if (j.contains("error"))
			{
				errors++;
				continue;
			}
			Rec r;
			for (int d = 0; d < 3; d++)
			{
				r.p[d] = j["p"][d];
				r.ijk_f[d] = j["ijk_f"][d];
				r.uvw[d] = j["uvw"][d];
				r.base[d] = j["base"][d];
				r.g[d] = j["g_index"][d];
			}
			for (int k = 0; k < 64; k++)
				r.stencil[k] = j["stencil"][k];
			for (int k = 0; k < 9; k++)
				r.h[k] = j["h_index"][k];
			r.x = j["x"];
			recs.push_back(r);
		}
		out("[%s] %zu samples read, %zu with a sampler error in the dump\n", label.c_str(), recs.size(), errors);
		if (recs.empty())
			return 2;
	}

	// --- (a) sampler on OpenVDB's stencils ------------------------------------
	{
		std::uint64_t max_ulp = 0, max_ulp_x = 0, max_ulp_g = 0, max_ulp_h = 0;
		size_t idx_mismatch = 0, bitwise = 0;
		const double inv_h = 1.0 / h;
		for (const Rec &r : recs)
		{
			double uvw[3];
			bool idx_ok = true;
			for (int d = 0; d < 3; d++)
			{
				const double x = r.p[d] * inv_h;
				const int b = int(std::floor(x));
				uvw[d] = x - double(b);
				idx_ok &= same_bits(x, r.ijk_f[d]) && b == r.base[d] && same_bits(uvw[d], r.uvw[d]);
			}
			idx_mismatch += !idx_ok;
			double o[10];
			gate6a::spline_hessian_ref(r.stencil, r.uvw, o);
			const double hv[9] = {o[4], o[5], o[6], o[5], o[7], o[8], o[6], o[8], o[9]};
			std::uint64_t ux = ulps(o[0], r.x), ug = 0, uh = 0;
			for (int d = 0; d < 3; d++)
				ug = std::max(ug, ulps(o[1 + d], r.g[d]));
			for (int k = 0; k < 9; k++)
				uh = std::max(uh, ulps(hv[k], r.h[k]));
			max_ulp_x = std::max(max_ulp_x, ux);
			max_ulp_g = std::max(max_ulp_g, ug);
			max_ulp_h = std::max(max_ulp_h, uh);
			max_ulp = std::max({max_ulp, ux, ug, uh});
			bitwise += (ux | ug | uh) == 0;
		}
		verdict(idx_mismatch == 0, "[" + label + "] a.index",
				fmt("p*(1/h), floor, uvw bitwise equal to OpenVDB worldToIndex at %zu/%zu samples", recs.size() - idx_mismatch, recs.size()));
		verdict(max_ulp <= 1, "[" + label + "] a.sampler",
				fmt("reference vs SplineSampler::sampleHessian on OpenVDB stencils: max %llu ULP (x %llu, g %llu, h %llu), %zu/%zu bitwise",
					(unsigned long long)max_ulp, (unsigned long long)max_ulp_x, (unsigned long long)max_ulp_g,
					(unsigned long long)max_ulp_h, bitwise, recs.size()));

		if (sdf_spline::kernel_available())
		{
			std::vector<double> st(64 * recs.size()), uv(3 * recs.size()), ko(10 * recs.size());
			for (size_t n = 0; n < recs.size(); n++)
			{
				std::memcpy(&st[64 * n], recs[n].stencil, sizeof recs[n].stencil);
				std::memcpy(&uv[3 * n], recs[n].uvw, sizeof recs[n].uvw);
			}
			sdf_spline::hessian_batch(st.data(), uv.data(), ko.data(), recs.size());
			std::uint64_t mk = 0;
			for (size_t n = 0; n < recs.size(); n++)
			{
				double o[10];
				gate6a::spline_hessian_ref(recs[n].stencil, recs[n].uvw, o);
				for (int k = 0; k < 10; k++)
					mk = std::max(mk, ulps(o[k], ko[10 * n + k]));
			}
			verdict(mk <= 1, "[" + label + "] a.kernel", fmt("Lean emit vs reference: max %llu ULP over %zu samples", (unsigned long long)mk, recs.size()));
		}
		else
			out("INFO [%s] a.kernel: pending (FIT_KERNELS_PENDING; lean/Fit/SdfSplineHessian.lean not emitted), reference only\n", label.c_str());
	}

	// --- (b) our grid vs OpenVDB ------------------------------------------------
	std::vector<std::array<double, 10>> ours(recs.size());
	{
		const auto t0 = std::chrono::steady_clock::now();
		SdfGrid grid(V, F, h);
		const double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		const auto t1 = std::chrono::steady_clock::now();
		for (size_t n = 0; n < recs.size(); n++)
			sample_ours(grid, recs[n].p, ours[n].data());
		const double pass_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
		const auto s = grid.stats();
		out("INFO [%s] grid: setup (AABB + winding BVH) %.3f s; %zu bricks (%zu voxels), %zu bytes (%.2f MB); "
			"%zu distance + %zu winding queries; fill %.3f s; sampling pass %.3f s (%.2f us/sample)\n",
			label.c_str(), setup_s, s.bricks, s.bricks * SdfGrid::kBrickVoxels, s.bytes, s.bytes / 1e6,
			s.distance_queries, s.winding_queries, s.fill_seconds, pass_s, 1e6 * pass_s / recs.size());

		// Voxel level: every distinct stencil voxel, ours vs OpenVDB's.
		std::unordered_map<std::uint64_t, double> vox;
		for (const Rec &r : recs)
			for (int a = 0; a < 4; a++)
				for (int b = 0; b < 4; b++)
					for (int c = 0; c < 4; c++)
					{
						const std::uint64_t k = (std::uint64_t(r.base[0] + a - 1 + (1 << 20)) << 42)
												| (std::uint64_t(r.base[1] + b - 1 + (1 << 20)) << 21)
												| std::uint64_t(r.base[2] + c - 1 + (1 << 20));
						vox.emplace(k, r.stencil[a * 16 + b * 4 + c]);
					}
		std::vector<double> dvox;
		size_t vnear = 0, vwithin = 0, vsign = 0, vpure = 0;
		for (const auto &[k, vv] : vox)
		{
			const int i = int(k >> 42) - (1 << 20), j = int((k >> 21) & 0x1FFFFF) - (1 << 20), kk = int(k & 0x1FFFFF) - (1 << 20);
			const double ov = grid.value(i, j, kk);
			vpure += same_bits(ov, grid.compute_voxel(i, j, kk));
			vsign += (ov > 0) == (vv > 0);
			if (std::abs(vv) < 5 * h)
			{
				vnear++;
				const double dv = std::abs(ov - vv) / h;
				dvox.push_back(dv);
				vwithin += dv <= 0.25;
			}
		}
		out("INFO [%s] voxels: %zu distinct stencil voxels, %zu with |v_vdb|<5 voxels: |dv|<=0.25 voxel %.4f%% "
			"(p50 %.4g p99 %.4g max %.4g voxel), sign agreement (all) %.4f%%\n",
			label.c_str(), vox.size(), vnear, 100.0 * vwithin / std::max<size_t>(vnear, 1), quantile(dvox, 0.5),
			quantile(dvox, 0.99), quantile(dvox, 1.0), 100.0 * vsign / vox.size());
		verdict(vpure == vox.size(), "[" + label + "] b.pure",
				fmt("brick value == value recomputed alone, bitwise, at %zu/%zu voxels", vpure, vox.size()));

		const Compare c = compare(recs, ours, h);
		report(("[" + label + "] b").c_str(), c);
		verdict(c.frac_within() >= 0.99, "[" + label + "] b.value", fmt("%.4f%% within 0.25 voxel (>= 99%%)", 100 * c.frac_within()));
		verdict(c.frac_sign() >= 0.995, "[" + label + "] b.sign", fmt("%.4f%% sign agreement (>= 99.5%%)", 100 * c.frac_sign()));
		verdict(c.frac_ang() >= 0.95, "[" + label + "] b.gradient", fmt("%.4f%% within 10 deg (>= 95%%)", 100 * c.frac_ang()));
	}

	// Fill order: a fresh grid walked in reverse gives bitwise the same samples.
	{
		SdfGrid grid(V, F, h);
		size_t same = 0;
		for (size_t n = recs.size(); n-- > 0;)
		{
			double o[10];
			sample_ours(grid, recs[n].p, o);
			bool eq = true;
			for (int k = 0; k < 10; k++)
				eq &= same_bits(o[k], ours[n][k]);
			same += eq;
		}
		verdict(same == recs.size(), "[" + label + "] b.order", fmt("reverse-order fill, bitwise equal samples %zu/%zu", same, recs.size()));
	}

	// Cache: same avatar and voxel give the same grid object; another voxel size does not.
	{
		auto g1 = SdfGrid::cached(V, F, h);
		auto g2 = SdfGrid::cached(V, F, h);
		auto g3 = SdfGrid::cached(V, F, 2 * h);
		auto g4 = SdfGrid::cached(V, F, h);
		verdict(g1 == g2 && g3 != g1 && g4 != g1 && g4->voxel_size() == h, "[" + label + "] cache",
				"cached(V,F,h) twice -> one grid; (V,F,2h) -> another; back to h -> rebuilt (one entry kept)");
		SdfGrid::clear_cache();
	}

	// --- (c) control: our grid shifted +0.5 voxel in x must fail (b) --------------
	{
		SdfGrid grid(V, F, h);
		std::vector<std::array<double, 10>> shifted(recs.size());
		for (size_t n = 0; n < recs.size(); n++)
		{
			const double q[3] = {recs[n].p[0] - 0.5 * h, recs[n].p[1], recs[n].p[2]};
			sample_ours(grid, q, shifted[n].data());
		}
		const Compare c = compare(recs, shifted, h);
		report(("[" + label + "] c (shift +0.5 voxel x)").c_str(), c);
		verdict(!c.pass(), "[" + label + "] c.control",
				fmt("shifted grid fails (b): value %.4f%% (>= 99%%: %s), sign %.4f%% (%s), gradient %.4f%% (%s)",
					100 * c.frac_within(), c.frac_within() >= 0.99 ? "pass" : "FAIL", 100 * c.frac_sign(),
					c.frac_sign() >= 0.995 ? "pass" : "FAIL", 100 * c.frac_ang(), c.frac_ang() >= 0.95 ? "pass" : "FAIL"));
	}

	out("[%s] SUMMARY PASS=%d FAIL=%d\n", label.c_str(), n_pass, n_fail);
	return n_fail ? 1 : 0;
}
