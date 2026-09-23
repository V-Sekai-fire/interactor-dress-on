// The bits behind the native_decide pins of lean/Fit/SdfSplineHessian.lean.
//
//   lean_fixtures            (built and run by gates/6-fit/sdf/run.sh)
//
// Two fixtures, built the same way as `fixtureStencil`, `fixture0` and
// `fixture1` in the Lean module: stencil value n is ((37 n) mod 64) / 7 - 4.5,
// uvw = (1/3, 2/7, 0.96875) and (0, 0.5, 5/9). For each it prints the ten
// outputs of the reference sampler (spline_ref.h) as IEEE bit patterns, in
// the Lean list syntax the pins use, and checks that the Lean emit
// (SdfSpline.cpp over kernels/fit/cpp) gives the same bits. Exit 1 on any
// difference.
#include <polyfem/solver/forms/garment_forms/SdfSpline.hpp>

#include "spline_ref.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{
	std::uint64_t bits(const double d)
	{
		std::uint64_t u;
		std::memcpy(&u, &d, sizeof u);
		return u;
	}
} // namespace

int main()
{
	using namespace polyfem::solver;
	double stencil[64];
	for (int n = 0; n < 64; n++)
		stencil[n] = double((37 * n) % 64) / 7 - 4.5;
	const double uvw[2][3] = {{1. / 3., 2. / 7., 0.96875}, {0., 0.5, 5. / 9.}};

	int bad = 0;
	for (int f = 0; f < 2; f++)
	{
		double ref[10], ker[10];
		gate6a::spline_hessian_ref(stencil, uvw[f], ref);
		if (sdf_spline::kernel_available())
			sdf_spline::hessian_batch(stencil, uvw[f], ker, 1);
		std::printf("fixture%d = [", f);
		for (int k = 0; k < 10; k++)
		{
			std::printf("%s0x%016" PRIX64, k ? ", " : "", bits(ref[k]));
			if (sdf_spline::kernel_available() && bits(ref[k]) != bits(ker[k]))
				bad++;
		}
		std::printf("]\n");
	}
	if (sdf_spline::kernel_available())
		std::printf("%s: Lean emit vs reference on both fixtures, %d of 20 outputs differ\n", bad ? "FAIL" : "PASS", bad);
	else
		std::printf("INFO: kernel left out (FIT_KERNELS_PENDING), reference bits only\n");
	return bad ? 1 : 0;
}
