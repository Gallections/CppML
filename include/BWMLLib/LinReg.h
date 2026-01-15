#pragma once

#include "NDArray.hpp"
#include <cmath>

namespace BWMLLib {
	class LinReg {

	private:
		double learning_rate;
		double convergence_tol;
		NDArray<double> weights;
		NDArray<double> biases;

	public:
		LinReg(double learning_rate, double convergence_tol = 1e-6);

		void initialize_parameters(int n_features);

		NDArray<double> forward(NDArray<double>& X);


	};
}