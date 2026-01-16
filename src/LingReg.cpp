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

	NDArray<double> BWMLLib::LinReg::forward(const NDArray<double>& X) const {
		return this->weights.batched_matmul(X) + this->biases;
	}

	double LinReg::compute_cost(NDArray<double> predictions) const {
		size_t m = predictions.get_size();
		double cost = (predictions - this->y).square().sum() / m;
		return cost;
	}

	void LinReg::backward(const NDArray<double> predictions) {
		size_t m = predictions.get_size();
		this->dw = (predictions - this->y).batched_matmul(this->X) / m;
		this->db = (predictions - this->y).sum() / m;
	}

	void LinReg::fit(NDArray<double>& X, NDArray<double>& y, size_t iterations) {
		this->X = std::move(X);
		this->y = std::move(y);
		initialize_parameters(this->X.get_shape()[1]);
		std::vector<double> costs;

		for (size_t i = 0; i < iterations; ++i) {
			NDArray<double> predictions = forward(this->X);
			double cost = compute_cost(predictions);
			this->weights = this->weights - this->dw * this->learning_rate;
			this->biases = this->biases - this->db * this->learning_rate;
			costs.push_back(cost);

			if (i % 100 == 0) {
				std::cout << "Iteration " << i << ", Cost "<< cost << std::endl;
			}

			if (i > 0 && abs(costs[costs.size() - 1]) - costs[costs.size() - 2] < convergence_tol) {
				std::cout << "Converged after "<< i << " iterations." << std::endl;
			}
		}
	}

	NDArray<double> LinReg::predict(NDArray<double>& X) const {
		NDArray<double> prediction = forward(X);
		return prediction;
	}

}