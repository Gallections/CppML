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
		NDArray<double> X;
		NDArray<double> y;
		NDArray<double> dw;
		NDArray<double> db;

	public:
		LinReg(double learning_rate, double convergence_tol = 1e-6);

		void initialize_parameters(int n_features);

		NDArray<double> forward(const NDArray<double>& X) const;

		double compute_cost(NDArray<double> predictions) const;

		void backward(const NDArray<double> predictions);

		void fit(NDArray<double>& X, NDArray<double>& y, size_t iterations);

		NDArray<double> predict(NDArray<double> &X) const;
	};
}