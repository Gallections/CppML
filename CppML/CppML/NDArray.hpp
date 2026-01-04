#pragma once

#include<iostream>
#include<vector>
#include<stdexcept>

template <typename T>
class NDArray {
private:
	// The flattened data
	std::vector<T> data;

	/*
		shape & strides are simply metadata for an ndarray:
		- `shape` is the representation of the number of elements in each dimension of the n-dimensional array.
		- `strides` is the number of memory steps that is required to access the next element in the current diemension.
		ex. given shape {2, 3, 4} - this is a 3D array with first dimension being 2 and last dimension being 4.
			[
				[
					[0, 0, 0, 0],
					[0, 0, 0, 0],
					[0, 0, 0, 0]
				],
				[
					[0, 0, 0, 0],
					[0, 0, 0, 0],
					[0, 0, 0, 0]
				]
			] <-- visualization of what a zero-initialized {2, 3, 4} ndarray looks like.

			The corresponding strides would be {12 ,4, 1}, this means for the last dimension, I only need to move one element
			in the data (falttened array) to find the next element in the across the last dimension, but I need to nove 4 elements
			at a time for the next element across the second dimension, and 3 * 4 = 12 elements across the first dimension.
	*/
	std::vector<size_t> shape;
	std::vector<size_t> strides;


	/*
		get_index() is a helper to get the falttened index from the ndarray representation.
	*/
	size_t get_index(const std::vector<size_t>& indices) const {
		if (indices.size() != shape.size()) {
			throw std::invalid_argument("Number of indices must match the dimensions!");
		}

		size_t flat_index = 0;
		for (size_t i = 0; i < shape.size(); ++i) {
			if (indices[i] >= shape[i]) {
				throw std::out_of_range("Index out of bounds!");
			}
			flat_index += indices[i] * strides[i];
		}

		return flat_index;
	}

public:

	// ==================== Constructor ======================
	NDArray(const std::vector<size_t>& shape_input) : shape(shape_input) {
		strides.resize(shape.size());
		size_t current_stride = 1;

		// We are comuting the strides based off the shape
		for (int i = shape.size() - 1; i >= 0; --i) {
			strides[i] = current_stride;
			current_stride *= shape[i];
		}

		// Resize Data
		size_t total_size = current_stride;  // By now, we know the current stride actually contains the total number of elements in the ndarray.
		data.resize(total_size, T());
	}

	// ====================== Accessor ========================
	T& operator()(const std::vector<size_t>& indices) {
		return data[get_index(indices)];
	}

	T operator()(const std::vector<size_t>& indices) const {
		return data[get_index(indices)];
	}

	// Getter for shape
	std::vector<size_t> get_shape() const {
		return shape;
	}

	// ===================== Math Operations ======================
	// ------- Elementwise Addition ---------
	NDArray<T> operator+(const NDArray<T>& other) const {
		if (shape != other.shape) {
			throw std::invalid_argument("The two ndarrays must have the same shape!");
		}

		NDArray<T> result(shape);
		for (size_t i = 0; i < data.size(); i++) {
			result[i] = data[i] + other.data[i];
		}
		return result;
	}

	// ----------- Check Equality ----------------
	bool operator==(const NDArray<T>& other) const {
		return (shape == other.shape) && (data == other.data);
	}

	// --------- Scalar Multiplication -------------
	NDArray<T> operator*(T scalar) const {
		NDArray<T> result(shape);
		for (size_t i = 0; i < data.size(); ++i) {
			result[i] = scalar * data[i];
		}
		return result;
	}

	// ----------------- Transpose -----------------
	NDArray<T> transpose(const NDArray& other) const {
		return NULL;
	};

	// ----------- Matrix Multiplication --------------
	/*
		We need to leverage an approach called batch matrix multiplication, which 
		takes the last 2 dimensions of the two ndarrays, and perform matrix multiplcations. 
		We have 2 pointers, each pointing to a 2D matrix, we perform matrix multiplication of 
		the 2 matrices, obtain the result and store it in a result ndarray. We then move each 
		pointers along their respective strides, and repeat the matrix multiplication process again
		& again.
	*/
	NDArray<T> matmul(const NDArray& other) const {
		 

	}


	/*
		Performs matrix multiplication, strictly restricsts the dimension of each array to be 2D.
	*/
	NDArray<T> matmul2D(const NDArray& other) const {
		if (shape.size() != 2 || other.shape.size() != 2) {
			throw std::invalid_argument("Each ndarray must be a matrix (2D NDArray)!");
		}
		if (shape[1] != other.shape[0]) {
			throw std::invalid_argument("The number of columns in your first matrix does not align with the number of rows in your second matrix!");
		}

		NDArray result(std::vector<size_t>{ shape[0], other.shape[1] }); // initialize the result matrix;

		for (size_t i = 0; i < shape[0]; ++i) {
			for (size_t j = 0; j < other.shape[1]; ++j) {
				T entry = T();
				for (size_t z = 0; z < shape[1]; ++z) {
					entry += data[i * strides[0] + z * strides[1]] * other.data[z * other.strides[0] + other.strides[1] * j];
				}
				result.data[i * result.strides[0] + j * result.strides[1]] = entry;
			}
		}
		return result;
	}


	// =================== Utility Functions ======================
	void print_data() const {
		std::cout << "[ ";
		for (size_t i = 0; i < data.size(); ++i) {
			std::cout << data[i] << " ";
		}
		std::cout << "]" << std::endl;
	}

	void print_shape() const{
		std::cout << "( ";
		for (size_t i = 0; i < shape.size(); i++) {
			std::cout << shape[i] << ", ";
		}
		std::cout << ")" << std::endl;
	}
};
