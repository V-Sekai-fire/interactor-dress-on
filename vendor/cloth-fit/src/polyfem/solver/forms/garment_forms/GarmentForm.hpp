#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <functional>

namespace polyfem::solver
{
	class AreaForm : public Form
	{
	public:
		AreaForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const double threshold = 1e-5) : V_(V), F_(F), threshold_(threshold) {}
		virtual ~AreaForm() = default;

		std::string name() const override { return "area"; }

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
		const double threshold_;
	};

	class DefGradForm : public Form
	{
	public:
		DefGradForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) : V_(V), F_(F) {}
		virtual ~DefGradForm() = default;

		std::string name() const override { return "deformation"; }

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
	};

	class AngleForm : public Form
	{
	public:
		AngleForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~AngleForm() = default;

		std::string name() const override { return "angle"; }

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd areas;
		Eigen::MatrixXd orig_angles;
	};


	class RelativeScalingForm : public Form
	{
	public:
		RelativeScalingForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~RelativeScalingForm() = default;

		std::string name() const override { return "relative-scaling"; }

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:

		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd orig_areas;
		Eigen::MatrixXd orig_dists;
	};


	class SimilarityForm : public Form
	{
	public:
		SimilarityForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~SimilarityForm() = default;

		std::string name() const override { return "similarity"; }

		// interactor-dress-on cut 6g-C: the Hessian assembly can be handed to
		// a hook (fit.elf's GPU path, guest/fit/fit_gpu.cpp). It receives the
		// solution, whether the blocks are to be projected to PSD, and the
		// output matrix; returning false falls back to the CPU path below,
		// which stays bitwise what it was when no hook is set.
		using HessianHook = std::function<bool(const Eigen::VectorXd &, bool, StiffnessMatrix &)>;
		void set_hessian_hook(HessianHook hook) { hessian_hook_ = std::move(hook); }

		// The hinges as the CPU path walks them, for a hook that assembles
		// the same blocks: hinge k of face i, local edge j is (F(i, le(j,0)),
		// F(i, le(j,1)), F(i, lv(j)), F(TT(i,j), lv(TTi(i,j)))) with le =
		// {{0,1},{1,2},{2,0}}, lv = {2,0,1}, k = 3 i + j, skipped when TT(i,j) < 0.
		const Eigen::MatrixXd &rest_vertices() const { return V_; }
		const Eigen::MatrixXi &faces() const { return F_; }
		const Eigen::MatrixXi &adjacency() const { return TT; }
		const Eigen::MatrixXi &adjacency_index() const { return TTi; }
		const Eigen::VectorXd &rest_areas() const { return orig_areas; }
		const Eigen::MatrixXd &rest_coeffs() const { return orig_coeffs; }

		// The CPU path's per-hinge blocks: similarity_hessian on x, projected
		// to PSD when asked, NOT weighted; 144 doubles per hinge, row-major
		// H(i, j) in the order of hinges above (skipped hinges left out). The
		// C1 gate's reference for the GPU blocks.
		void hessian_blocks(const Eigen::VectorXd &x, bool project_psd, std::vector<double> &blocks) const;

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:

		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd orig_areas;
		Eigen::MatrixXd orig_coeffs;
		HessianHook hessian_hook_;
	};

	class NormalForm : public Form
	{
	public:
		NormalForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~NormalForm() = default;

		std::string name() const override { return "normal"; }

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;

		Eigen::VectorXd orig_areas;
	};

	// class GlobalPositionalForm : public Form
	// {
	// public:
	// 	GlobalPositionalForm(
	// 		const Eigen::MatrixXd &V,
	// 		const Eigen::MatrixXi &F,
	// 		const Eigen::MatrixXd &source_skeleton_v,
	// 		const Eigen::MatrixXd &target_skeleton_v,
	// 		const Eigen::MatrixXi &skeleton_edges,
    //     	const Eigen::MatrixXd &skin_weights);
	// 	virtual ~GlobalPositionalForm() = default;

	// 	std::string name() const override { return "global-relative-position"; }

	// protected:
	// 	/// @brief Compute the potential value
	// 	/// @param x Current solution
	// 	/// @return Value of the contact barrier potential
	// 	double value_unweighted(const Eigen::VectorXd &x) const override;

	// 	/// @brief Compute the first derivative of the value wrt x
	// 	/// @param[in] x Current solution
	// 	/// @param[out] gradv Output gradient of the value wrt x
	// 	void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	// 	/// @brief Compute the second derivative of the value wrt x
	// 	/// @param x Current solution
	// 	/// @param hessian Output Hessian of the value wrt x
	// 	void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	// private:
	// 	const Eigen::MatrixXd V_;

	// 	const Eigen::MatrixXd source_skeleton_v_;
	// 	const Eigen::MatrixXd target_skeleton_v_;
	// 	const Eigen::MatrixXi skeleton_edges_;
	// 	const Eigen::MatrixXd skin_weights_;

	// 	Eigen::VectorXi bones;
	// 	Eigen::VectorXd relative_positions;
	// };
} // namespace polyfem::solver
