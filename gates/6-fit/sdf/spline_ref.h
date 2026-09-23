// Gate 6a reference: the tricubic B-spline value/gradient/Hessian of the
// Huangzizhou/openvdb fork (c96cb069, openvdb/tools/Interpolation.h,
// SplineSampler::spline*, trilinearInterpolationHessian), transcribed
// operation for operation so that it rounds the same way.
//
// Used ONLY by the native Gate 6a test to cross-check the Lean kernel
// (lean/Fit/SdfSplineHessian.lean -> kernels/fit/cpp/sdf_spline_hessian_emit.cpp)
// and OpenVDB's own sampler. It is not compiled into polyfem or fit.elf
// (AGENTS.md rule 2: the shipped sampler is the Lean emit).
//
// Output layout is the kernel's (SdfSpline.hpp): x, gx, gy, gz, hxx, hxy,
// hxz, hyy, hyz, hzz, index space.
#pragma once

#include <cmath>

namespace gate6a
{
	inline double spline(const double x)
	{
		const double absx = std::abs(x);
		if (absx >= 2)
			return 0.;
		if (absx >= 1)
		{
			const double tmp = 2. - absx;
			return tmp * tmp * tmp / 6.;
		}
		return 2. / 3. + (0.5 * absx - 1) * x * x;
	}

	inline double spline_1st_deriv(const double x)
	{
		const double absx = std::abs(x);
		if (absx >= 2)
			return 0.;
		if (absx >= 1)
		{
			const double tmp = 2. - absx;
			return -0.5 * tmp * tmp * (x > 0 ? 1 : -1);
		}
		return x * (1.5 * absx - 2);
	}

	inline double spline_2nd_deriv(const double x)
	{
		const double absx = std::abs(x);
		if (absx >= 2)
			return 0.;
		if (absx >= 1)
			return 2. - absx;
		return 3 * absx - 2;
	}

	// data: 64 values, data[i][j][k] at i*16 + j*4 + k. uvw: index-space fraction.
	inline void spline_hessian_ref(const double *data, const double *uvw, double *out)
	{
		double basis[3][4], deriv1[3][4], deriv2[3][4];
		for (int d = 0; d < 3; d++)
		{
			basis[d][0] = spline(uvw[d] + 1);
			deriv1[d][0] = spline_1st_deriv(uvw[d] + 1);
			deriv2[d][0] = spline_2nd_deriv(uvw[d] + 1);

			basis[d][1] = spline(uvw[d]);
			deriv1[d][1] = spline_1st_deriv(uvw[d]);
			deriv2[d][1] = spline_2nd_deriv(uvw[d]);

			basis[d][2] = spline(1 - uvw[d]);
			deriv1[d][2] = -spline_1st_deriv(1 - uvw[d]);
			deriv2[d][2] = spline_2nd_deriv(1 - uvw[d]);

			basis[d][3] = spline(2 - uvw[d]);
			deriv1[d][3] = -spline_1st_deriv(2 - uvw[d]);
			deriv2[d][3] = spline_2nd_deriv(2 - uvw[d]);
		}

		double x = 0, g0 = 0, g1 = 0, g2 = 0;
		double h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
				{
					const double v = data[i * 16 + j * 4 + k];
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
} // namespace gate6a
