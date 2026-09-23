#include "fit_df.hpp"

#include "g6g.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace g6g::fit {
namespace {

struct Sample {
	DF x;
	DF g[3];
	DF h[3][3];
};

std::unordered_map<const void *, std::vector<Sample>> g_samples;

// The double path reads totalP, sized and zero-filled at construction, so a
// form that is evaluated before its first solution_changed (the reduced phase
// enables the fit form and asks its value before the solver's first
// solution_changed) sees x = 0, g = 0, h = 0. Absent df32 samples read the
// same; a stale vector from a form that lived at this address before is
// erased by reset() from the constructor.
const std::vector<Sample> &samples_of(const Setup &s)
{
	static const std::vector<Sample> none;
	auto it = g_samples.find(s.key);
	if (it == g_samples.end() || it->second.size() < size_t(s.F_.rows()) * 15)
		return none;
	return it->second;
}
inline const Sample &sample_at(const std::vector<Sample> &sp, size_t i)
{
	static const Sample zero{};
	return sp.empty() ? zero : sp[i];
}

bool sdf_f32()
{
	static const bool v = [] {
		const char *e = std::getenv("G6G_SDF_F32");
		return e && std::strcmp(e, "1") == 0;
	}();
	return v;
}

// gates/6-fit/sdf/spline_ref.h (the Lean kernel's arithmetic), in T.
template <class T>
T spline(const T &x)
{
	using std::abs;
	const T absx = abs(x);
	if (absx >= T(2))
		return T(0.);
	if (absx >= T(1)) {
		const T tmp = T(2.) - absx;
		return tmp * tmp * tmp / T(6.);
	}
	return T(2. / 3.) + (T(0.5) * absx - T(1)) * x * x;
}
template <class T>
T spline_1st_deriv(const T &x)
{
	using std::abs;
	const T absx = abs(x);
	if (absx >= T(2))
		return T(0.);
	if (absx >= T(1)) {
		const T tmp = T(2.) - absx;
		return T(-0.5) * tmp * tmp * T(x > T(0) ? 1 : -1);
	}
	return x * (T(1.5) * absx - T(2));
}
template <class T>
T spline_2nd_deriv(const T &x)
{
	using std::abs;
	const T absx = abs(x);
	if (absx >= T(2))
		return T(0.);
	if (absx >= T(1))
		return T(2.) - absx;
	return T(3) * absx - T(2);
}

template <class T>
void spline_hessian(const T *data, const T *uvw, T *out)
{
	T basis[3][4], deriv1[3][4], deriv2[3][4];
	for (int d = 0; d < 3; d++) {
		basis[d][0] = spline(uvw[d] + T(1));
		deriv1[d][0] = spline_1st_deriv(uvw[d] + T(1));
		deriv2[d][0] = spline_2nd_deriv(uvw[d] + T(1));
		basis[d][1] = spline(uvw[d]);
		deriv1[d][1] = spline_1st_deriv(uvw[d]);
		deriv2[d][1] = spline_2nd_deriv(uvw[d]);
		basis[d][2] = spline(T(1) - uvw[d]);
		deriv1[d][2] = -spline_1st_deriv(T(1) - uvw[d]);
		deriv2[d][2] = spline_2nd_deriv(T(1) - uvw[d]);
		basis[d][3] = spline(T(2) - uvw[d]);
		deriv1[d][3] = -spline_1st_deriv(T(2) - uvw[d]);
		deriv2[d][3] = spline_2nd_deriv(T(2) - uvw[d]);
	}
	T x = T(0), g0 = T(0), g1 = T(0), g2 = T(0);
	T h00 = T(0), h01 = T(0), h02 = T(0), h11 = T(0), h12 = T(0), h22 = T(0);
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			for (int k = 0; k < 4; k++) {
				const T v = data[i * 16 + j * 4 + k];
				x += v * basis[0][i] * basis[1][j] * basis[2][k];
				g0 += v * deriv1[0][i] * basis[1][j] * basis[2][k];
				g1 += v * basis[0][i] * deriv1[1][j] * basis[2][k];
				g2 += v * basis[0][i] * basis[1][j] * deriv1[2][k];
				h00 += v * deriv2[0][i] * basis[1][j] * basis[2][k];
				h01 += v * deriv1[0][i] * deriv1[1][j] * basis[2][k];
				h02 += v * deriv1[0][i] * basis[1][j] * deriv1[2][k];
				h11 += v * basis[0][i] * deriv2[1][j] * basis[2][k];
				h12 += v * basis[0][i] * deriv1[1][j] * deriv1[2][k];
				h22 += v * basis[0][i] * basis[1][j] * deriv2[2][k];
			}
	out[0] = x;
	out[1] = g0;
	out[2] = g1;
	out[3] = g2;
	out[4] = h00;
	out[5] = h01;
	out[6] = h02;
	out[7] = h11;
	out[8] = h12;
	out[9] = h22;
}

DF face_area(const Setup &s, int f)
{
	using V3 = Eigen::Matrix<DF, 3, 1>;
	V3 a, b, c;
	for (int d = 0; d < 3; d++) {
		a(d) = DF(s.V_(s.F_(f, 0), d));
		b(d) = DF(s.V_(s.F_(f, 1), d));
		c(d) = DF(s.V_(s.F_(f, 2), d));
	}
	return (b - a).cross(c - a).norm() / DF(2);
}

} // namespace

void reset(const void *key) { g_samples.erase(key); }

void solution_changed(const Setup &s, const polyfem::solver::SdfGrid &grid, const Eigen::VectorXd &x,
					  const std::vector<polyfem::solver::SdfHess> &ref)
{
	CtxScope ctx(form_ctx(F_FIT));
	const bool cmp = comparing();
	const int n = 15;
	std::vector<Sample> &out = g_samples[s.key];
	out.assign(size_t(s.F_.rows()) * n, Sample{});
	const DF inv_h = DF(1.0 / grid.voxel_size());
	const DF vox = DF(s.voxel);
	const DF inv_h2 = DF(1.) / vox / vox;
	double stencil_d[64];
	DF stencil[64];
	for (const int f : s.faces) {
		DF M[3][3];
		for (int v = 0; v < 3; v++)
			for (int d = 0; d < 3; d++)
				M[v][d] = DF(x(s.F_(f, v) * 3 + d)) + DF(s.V_(s.F_(f, v), d));
		for (int i = 0; i < n; i++) {
			DF p[3], uvw[3];
			std::array<int, 3> base;
			for (int d = 0; d < 3; d++) {
				p[d] = (DF(s.P(i, 0)) * M[0][d] + DF(s.P(i, 1)) * M[1][d]) + DF(s.P(i, 2)) * M[2][d];
				const DF xi = p[d] * inv_h;
				base[d] = int(std::floor(xi.v));
				uvw[d] = xi - DF(double(base[d]));
			}
			grid.stencil(base, stencil_d);
			for (int k = 0; k < 64; k++)
				stencil[k] = sdf_f32() ? DF(double(float(stencil_d[k]))) : DF(stencil_d[k]);
			DF o[10];
			spline_hessian<DF>(stencil, uvw, o);
			Sample &sm = out[size_t(f) * n + i];
			sm.x = o[0];
			for (int d = 0; d < 3; d++)
				sm.g[d] = o[1 + d] / vox;
			const DF hh[3][3] = {{o[4], o[5], o[6]}, {o[5], o[7], o[8]}, {o[6], o[8], o[9]}};
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					sm.h[a][b] = hh[a][b] * inv_h2;
			if (cmp) {
				CtxScope none(CTX_NONE);
				const polyfem::solver::SdfHess &r = ref[size_t(f) * n + i];
				g_form[F_FITS].elem[Q_VALUE].add(rel_err(r.x, sm.x.v));
				const double gd[3] = {r.g(0), r.g(1), r.g(2)};
				const double gf[3] = {sm.g[0].v, sm.g[1].v, sm.g[2].v};
				g_form[F_FITS].elem[Q_GRAD].add(rel_err_vec(gd, gf, 3));
				double hd[9], hf[9];
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++) {
						hd[3 * a + b] = r.h(a, b);
						hf[3 * a + b] = sm.h[a][b].v;
					}
				g_form[F_FITS].elem[Q_HESS].add(rel_err_vec(hd, hf, 9));
			}
		}
	}
}

double value(const Setup &s)
{
	const std::vector<Sample> &sp = samples_of(s);
	const int n = 15;
	const bool cmp = comparing();
	DF val = DF(0);
	for (const int f : s.faces) {
		const DF area = face_area(s, f);
		for (int i = 0; i < n; i++) {
			const DF tmp = sample_at(sp, size_t(f) * n + i).x;
			const DF init = DF(s.initial_distance(f * n + i));
			if (tmp > init) {
				const DF e = area * DF(s.weights(i)) * pow(tmp - init, s.power);
				val += e;
			}
		}
	}
	(void)cmp;
	return val.v;
}

void gradient(const Setup &s, Eigen::VectorXd &gradv)
{
	const std::vector<Sample> &sp = samples_of(s);
	const int n = 15;
	std::vector<DF> g(size_t(s.V_.rows()) * 3, DF(0));
	for (const int f : s.faces) {
		const DF area = face_area(s, f);
		for (int i = 0; i < n; i++) {
			const Sample &t = sample_at(sp, size_t(f) * n + i);
			const DF init = DF(s.initial_distance(f * n + i));
			if (!(t.x > init))
				continue;
			for (int d = 0; d < 3; d++) {
				const DF c = pow(t.x - init, s.power - 1) * DF(s.power) * area * DF(s.weights(i)) * t.g[d];
				for (int v = 0; v < 3; v++)
					g[size_t(s.F_(f, v)) * 3 + d] += c * DF(s.P(i, v));
			}
		}
	}
	gradv.resize(Eigen::Index(g.size()));
	for (size_t i = 0; i < g.size(); i++)
		gradv(Eigen::Index(i)) = g[i].v;
}

void hessian(const Setup &s, Eigen::SparseMatrix<double> &hessian)
{
	const std::vector<Sample> &sp = samples_of(s);
	const int n = 15;
	std::vector<Eigen::Triplet<double>> trip;
	for (const int f : s.faces) {
		const DF area = face_area(s, f);
		DF lh[9][9];
		for (auto &r : lh)
			for (auto &c : r)
				c = DF(0);
		for (int i = 0; i < n; i++) {
			const Sample &t = sample_at(sp, size_t(f) * n + i);
			const DF init = DF(s.initial_distance(f * n + i));
			if (t.x <= init)
				continue;
			DF h[3][3];
			const DF c1 = pow(t.x - init, s.power - 1) * DF(s.power);
			const DF c2 = pow(t.x - init, s.power - 2) * DF(s.power) * DF(s.power - 1);
			const DF aw = area * DF(s.weights(i));
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					h[a][b] = ((t.h[a][b] * c1) + (t.g[a] * t.g[b]) * c2) * aw;
			for (int d = 0; d < 3; d++)
				for (int k = 0; k < 3; k++)
					for (int a = 0; a < 3; a++)
						for (int l = 0; l < 3; l++)
							lh[a * 3 + d][l * 3 + k] += DF(s.P(i, a)) * DF(s.P(i, l)) * h[d][k];
		}
		for (int d = 0; d < 3; d++)
			for (int k = 0; k < 3; k++)
				for (int a = 0; a < 3; a++)
					for (int l = 0; l < 3; l++)
						trip.emplace_back(s.F_(f, a) * 3 + d, s.F_(f, l) * 3 + k, lh[a * 3 + d][l * 3 + k].v);
	}
	const Eigen::Index N = s.V_.rows() * 3;
	hessian.resize(N, N);
	hessian.setFromTriplets(trip.begin(), trip.end());
}

} // namespace g6g::fit
