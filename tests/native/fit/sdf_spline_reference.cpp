// fit_native only (FIT_SDF=sdfgrid, FIT_SDF_SAMPLER=reference): the
// sdf_spline::hessian_batch entry point of SdfSpline.hpp backed by the Gate 6a
// reference sampler (gates/6-fit/sdf/spline_ref.h, the Huangzizhou/openvdb
// SplineSampler transcribed operation for operation; 0 ULP against OpenVDB's
// sampleHessian on 99,315 samples).
//
// It stands in for vendor/cloth-fit/.../SdfSpline.cpp so the SdfGrid FitForm
// can be run end to end natively before lean/Fit/SdfSplineHessian.lean is
// emitted. It is never compiled into polyfem or fit.elf (AGENTS.md rule 2: the
// shipped sampler is the Lean emit); tests/native/fit/CMakeLists.txt picks this
// file or SdfSpline.cpp, never both.

#include <polyfem/solver/forms/garment_forms/SdfSpline.hpp>

#include "../../../gates/6-fit/sdf/spline_ref.h"

namespace polyfem::solver::sdf_spline
{
	bool kernel_available() { return true; }

	void hessian_batch(const double *stencil, const double *uvw, double *out, const std::size_t n)
	{
		for (std::size_t s = 0; s < n; ++s)
			gate6a::spline_hessian_ref(stencil + kStencil * s, uvw + 3 * s, out + kOut * s);
	}
} // namespace polyfem::solver::sdf_spline
