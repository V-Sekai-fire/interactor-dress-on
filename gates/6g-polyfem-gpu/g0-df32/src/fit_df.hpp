// FitForm<4> (cloth-fit FitForm.cpp, SdfGrid + the Lean tricubic sampler) in
// the df32 model. solution_changed samples the SDF at the 15 points of every
// fit face in df32 (sample position P*M, index-space fraction, the tricubic
// B-spline value/gradient/Hessian in the kernel's order acc += ((data*a)*b)*c,
// the 1/h and 1/h^2 scalings); value/gradient/Hessian then run in df32 on those
// samples. The grid voxel values are stored df32 (G6G_SDF_F32=1: float32).
// Compare mode records the sampler's error per sample (form "fit-sampler").
#pragma once

#include <polyfem/solver/forms/garment_forms/SdfGrid.hpp>
#include <polyfem/solver/forms/garment_forms/SdfSpline.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <vector>

namespace g6g::fit {

struct Setup {
	const void *key;
	const Eigen::MatrixXd &V_;
	const Eigen::MatrixXi &F_;
	const std::vector<int> &faces;
	const Eigen::Matrix<double, 15, 3> &P;
	const Eigen::Matrix<double, 15, 1> &weights;
	const Eigen::VectorXd &initial_distance;
	int power;
	double voxel;
};

/// FitForm constructor: forget any samples a form at this address left.
void reset(const void *key);

/// After the original solution_changed: sample in df32 (and compare against
/// the double samples `ref`, 15 per face, when comparing).
void solution_changed(const Setup &s, const polyfem::solver::SdfGrid &grid, const Eigen::VectorXd &x,
					  const std::vector<polyfem::solver::SdfHess> &ref);
double value(const Setup &s);
void gradient(const Setup &s, Eigen::VectorXd &gradv);
void hessian(const Setup &s, Eigen::SparseMatrix<double> &hessian);

} // namespace g6g::fit
