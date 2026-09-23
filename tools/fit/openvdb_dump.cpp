// openvdb_dump: rebuild cloth-fit FitForm's SDF grid from a run's
// target_avatar.obj and dump grid values, the 4x4x4 spline stencils and
// SplineSampler::sampleHessian at given points. Oracle for Gate 6a.
//
// The grid is built exactly as FitForm<n>::FitForm does (FitForm.cpp:114-129):
// linear transform of voxel_size, points rounded to float (Vec3s),
// meshToSignedDistanceField<DoubleGrid>(xform, points, tris, {}, 150, 1).
// target_avatar.obj holds gstate.avatar_v written with fmt "{}" (shortest
// round-trip) and is parsed here with strtod, so the doubles match; its faces
// are missing (see --faces), so they come from the input avatar.obj.
//
// Usage:
//   openvdb_dump <target_avatar.obj> [options]
//     --faces <obj>        take triangles from this OBJ (the input avatar.obj):
//                          upstream's target_avatar.obj carries vertices only
//                          (eigen_to_obj_data makes no object, so the writer
//                          drops every face); avatar_f is the input's faces,
//                          fan-triangulated in file order (optimize.cpp:1103)
//     --voxel <h>          voxel size (default 0.01, setup.json voxel_size)
//     --points <file>      world-space points, "x y z" per line
//     --obj-verts <file>   use the vertices of an OBJ as points
//     --random <N>         N points uniform in the avatar bbox grown by --margin
//     --margin <m>         (default 0.1)
//     --seed <s>           mt19937_64 seed for --random (default 1)
//     --out <file>         JSON lines output (default stdout)
//     --active <file>      also write every active voxel: int32 i,j,k + f64 value
//     --iso <file>         also write volumeToMesh(grid, 0) as OBJ, the same call
//                          FitForm makes for the run's sdf.obj, so the rebuilt
//                          grid can be checked against the run's grid
//
// Output: line 1 is a header object (grid stats, build time); then one object
// per point: p (world), ijk_f (index-space coordinate), base (floor), uvw,
// stencil (64 values, data[i][j][k] with i,j,k = offsets -1..2 from base,
// flattened i-major exactly as SplineSampler::getValues fills it), value at
// base voxel, and the sampler's x, g (3), h (9, row-major) in index space and
// scaled to world as FitForm::solution_changed does (g/h, h/h^2).
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;

namespace
{
	struct Mesh
	{
		std::vector<double> v; // xyz
		std::vector<int> f;    // tri indices
	};

	bool read_obj(const std::string &path, Mesh &m)
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
					m.v.push_back(std::strtod(s, &e));
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
					int idx = std::atoi(tok.c_str()); // stops at '/'
					poly.push_back(idx > 0 ? idx - 1 : int(m.v.size() / 3) + idx);
				}
				for (size_t k = 1; k + 1 < poly.size(); k++)
				{
					m.f.push_back(poly[0]);
					m.f.push_back(poly[k]);
					m.f.push_back(poly[k + 1]);
				}
			}
		}
		return true;
	}

	void die(const std::string &msg)
	{
		std::fprintf(stderr, "openvdb_dump: %s\n", msg.c_str());
		std::exit(2);
	}
} // namespace

int main(int argc, char **argv)
{
	if (argc < 2)
		die("usage: openvdb_dump <target_avatar.obj> [--voxel h] [--points f] [--obj-verts f] [--random N] [--margin m] [--seed s] [--out f] [--active f]");

	std::string avatar_path = argv[1];
	double voxel = 0.01, margin = 0.1;
	std::string points_path, objv_path, out_path, active_path, faces_path, iso_path;
	long n_random = 0;
	unsigned long long seed = 1;
	for (int a = 2; a < argc; a++)
	{
		std::string k = argv[a];
		auto next = [&]() -> std::string {
			if (a + 1 >= argc)
				die("missing value for " + k);
			return argv[++a];
		};
		if (k == "--voxel") voxel = std::strtod(next().c_str(), nullptr);
		else if (k == "--points") points_path = next();
		else if (k == "--faces") faces_path = next();
		else if (k == "--obj-verts") objv_path = next();
		else if (k == "--random") n_random = std::strtol(next().c_str(), nullptr, 10);
		else if (k == "--margin") margin = std::strtod(next().c_str(), nullptr);
		else if (k == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
		else if (k == "--out") out_path = next();
		else if (k == "--active") active_path = next();
		else if (k == "--iso") iso_path = next();
		else die("unknown option " + k);
	}

	Mesh avatar;
	if (!read_obj(avatar_path, avatar) || avatar.v.empty())
		die("cannot read vertices from " + avatar_path);
	if (!faces_path.empty())
	{
		Mesh fm;
		if (!read_obj(faces_path, fm) || fm.f.empty())
			die("cannot read triangles from " + faces_path);
		if (fm.v.size() != avatar.v.size())
			die("vertex count mismatch between " + avatar_path + " and " + faces_path);
		avatar.f = fm.f;
	}
	if (avatar.f.empty())
		die("no triangles: pass --faces <input avatar.obj>");

	openvdb::initialize();

	// --- FitForm.cpp:116-129, verbatim in effect -----------------------------
	math::Transform::Ptr xform = math::Transform::createLinearTransform(voxel);
	std::vector<Vec3s> points;
	std::vector<Vec3I> triangles;
	std::vector<Vec4I> quads;
	points.reserve(avatar.v.size() / 3);
	for (size_t i = 0; i < avatar.v.size() / 3; i++)
		points.push_back(Vec3s(avatar.v[3 * i], avatar.v[3 * i + 1], avatar.v[3 * i + 2]));
	for (size_t i = 0; i < avatar.f.size() / 3; i++)
		triangles.push_back(Vec3I(avatar.f[3 * i], avatar.f[3 * i + 1], avatar.f[3 * i + 2]));

	const auto t0 = std::chrono::steady_clock::now();
	DoubleGrid::Ptr grid = tools::meshToSignedDistanceField<DoubleGrid>(*xform, points, triangles, quads, 150, 1);
	const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	// ---------------------------------------------------------------------------

	std::ofstream fout;
	std::ostream *out = &std::cout;
	if (!out_path.empty())
	{
		fout.open(out_path);
		if (!fout)
			die("cannot open " + out_path);
		out = &fout;
	}

	const CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
	double bmin[3] = {1e300, 1e300, 1e300}, bmax[3] = {-1e300, -1e300, -1e300};
	for (size_t i = 0; i < avatar.v.size() / 3; i++)
		for (int d = 0; d < 3; d++)
		{
			bmin[d] = std::min(bmin[d], avatar.v[3 * i + d]);
			bmax[d] = std::max(bmax[d], avatar.v[3 * i + d]);
		}

	nlohmann::json hdr;
	hdr["avatar"] = avatar_path;
	hdr["faces_from"] = faces_path.empty() ? avatar_path : faces_path;
	hdr["n_verts"] = avatar.v.size() / 3;
	hdr["n_tris"] = avatar.f.size() / 3;
	hdr["voxel_size"] = voxel;
	hdr["ex_band"] = 150;
	hdr["in_band"] = 1;
	hdr["background"] = grid->background();
	hdr["active_voxels"] = grid->activeVoxelCount();
	hdr["leaf_nodes"] = grid->tree().leafCount();
	hdr["mem_bytes"] = grid->memUsage();
	hdr["active_bbox_min"] = {bbox.min().x(), bbox.min().y(), bbox.min().z()};
	hdr["active_bbox_max"] = {bbox.max().x(), bbox.max().y(), bbox.max().z()};
	hdr["avatar_bbox_min"] = {bmin[0], bmin[1], bmin[2]};
	hdr["avatar_bbox_max"] = {bmax[0], bmax[1], bmax[2]};
	hdr["build_seconds"] = build_s;
	hdr["stencil_order"] = "data[i][j][k], i,j,k = -1..2 offsets from floor(ijk_f) along x,y,z; flattened i*16+j*4+k";
	*out << hdr.dump() << "\n";

	if (!active_path.empty())
	{
		FILE *fa = std::fopen(active_path.c_str(), "wb");
		if (!fa)
			die("cannot open " + active_path);
		for (auto it = grid->cbeginValueOn(); it; ++it)
		{
			if (it.isVoxelValue())
			{
				const Coord c = it.getCoord();
				const int32_t ijk[3] = {c.x(), c.y(), c.z()};
				const double val = *it;
				std::fwrite(ijk, sizeof(int32_t), 3, fa);
				std::fwrite(&val, sizeof(double), 1, fa);
			}
			else
			{
				// Active tile: expand every voxel it covers.
				CoordBBox tb;
				it.getBoundingBox(tb);
				const double val = *it;
				for (auto c = tb.begin(); c; ++c)
				{
					const int32_t ijk[3] = {(*c).x(), (*c).y(), (*c).z()};
					std::fwrite(ijk, sizeof(int32_t), 3, fa);
					std::fwrite(&val, sizeof(double), 1, fa);
				}
			}
		}
		std::fclose(fa);
	}

	if (!iso_path.empty())
	{
		std::vector<Vec3s> ipts;
		std::vector<Vec4I> iquads;
		tools::volumeToMesh(*grid, ipts, iquads, 0.);
		FILE *fi = std::fopen(iso_path.c_str(), "w");
		if (!fi)
			die("cannot open " + iso_path);
		for (const auto &v : ipts)
			std::fprintf(fi, "v %.9g %.9g %.9g\n", double(v(0)), double(v(1)), double(v(2)));
		for (const auto &f : iquads)
		{
			std::fprintf(fi, "f %u %u %u\n", f(0) + 1, f(1) + 1, f(2) + 1);
			std::fprintf(fi, "f %u %u %u\n", f(0) + 1, f(2) + 1, f(3) + 1);
		}
		std::fclose(fi);
	}

	std::vector<double> q;
	if (!points_path.empty())
	{
		std::ifstream in(points_path);
		if (!in)
			die("cannot open " + points_path);
		double x, y, z;
		while (in >> x >> y >> z)
		{
			q.push_back(x);
			q.push_back(y);
			q.push_back(z);
		}
	}
	if (!objv_path.empty())
	{
		Mesh m;
		if (!read_obj(objv_path, m))
			die("cannot read " + objv_path);
		q.insert(q.end(), m.v.begin(), m.v.end());
	}
	if (n_random > 0)
	{
		std::mt19937_64 rng(seed);
		for (long n = 0; n < n_random; n++)
			for (int d = 0; d < 3; d++)
			{
				const double lo = bmin[d] - margin, hi = bmax[d] + margin;
				// Explicit mapping, not uniform_real_distribution (implementation-defined).
				const double u = double(rng() >> 11) * 0x1.0p-53;
				q.push_back(lo + (hi - lo) * u);
			}
	}

	DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
	for (size_t n = 0; n < q.size() / 3; n++)
	{
		const math::Vec3<double> p(q[3 * n], q[3 * n + 1], q[3 * n + 2]);
		const Vec3R ijk_f = grid->transformPtr()->worldToIndex(p);
		const Vec3i base = tools::local_util::floorVec3(ijk_f);
		const Vec3R uvw = ijk_f - base;

		nlohmann::json r;
		r["i"] = n;
		r["p"] = {p[0], p[1], p[2]};
		r["ijk_f"] = {ijk_f[0], ijk_f[1], ijk_f[2]};
		r["base"] = {base[0], base[1], base[2]};
		r["uvw"] = {uvw[0], uvw[1], uvw[2]};
		r["value_at_base"] = acc.getValue(Coord(base));

		double data[4][4][4];
		bool ok = true;
		try
		{
			tools::SplineSampler::getValues(data, acc, Coord(base));
		}
		catch (const std::exception &e)
		{
			ok = false;
			r["error"] = e.what();
		}
		if (ok)
		{
			std::vector<double> st;
			st.reserve(64);
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					for (int k = 0; k < 4; k++)
						st.push_back(data[i][j][k]);
			r["stencil"] = st;

			// The exact call FitForm::solution_changed makes.
			const tools::HessType<double> H = tools::SplineSampler::sampleHessian(acc, ijk_f);
			r["x"] = H.x;
			r["g_index"] = {H.g[0], H.g[1], H.g[2]};
			std::vector<double> h9;
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					h9.push_back(H.h(a, b));
			r["h_index"] = h9;
			r["g_world"] = {H.g[0] / voxel, H.g[1] / voxel, H.g[2] / voxel};
			std::vector<double> hw;
			for (double v : h9)
				hw.push_back(v * (1. / voxel / voxel));
			r["h_world"] = hw;
		}
		*out << r.dump() << "\n";
	}
	return 0;
}
