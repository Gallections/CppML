#include "BWMLLib/LinReg.h"
#include <cmath>
#include <vector>
#include "NDArray.hpp"

namespace BWMLLib {
	/*
		Implementation of Linear Regression algorithm
	*/

	LinReg::LinReg(double learning_rate, double convergence_tol) {
		this->learning_rate = learning_rate;
		this->convergence_tol = convergence_tol;
	}

	void LinReg::initialize_parameters(int n_features) {
		this->biases = NDArray<double>({ 1 });
		this->weights = NDArray<double>({ static_cast<size_t>(n_features) }); // default initializes to zero.
	}

	NDArray<double> LinReg::forward(NDArray<double>& X) {
		return this->weights.batched_matmul(X) + this->biases;
	}


}