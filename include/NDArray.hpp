#pragma once

#include<iostream>
#include<vector>
#include<stdexcept>
#include<format>

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

	/*
		This is an internal function used for computing matrix multiplcation directly at memory address. 
		The goal is to use this as a helper function for batched matrix multiplcation (high dimensional tensor multiplcation).
		This function should directly modify the value at C_ptr.
		
		Params: 
			A_ptr: a pointer to the first matrix (A)
			B_ptr: a pointer to the second matrix (B)
			C_ptr: a pointer to the location to store the matrix product of A@B
			M: the # of rows of A@B, 
			N: the # of cols of A@B
			K: the matching # of cols/rows in A@B.
		
	*/
	static void _matmul(const T* A_ptr, const T* B_ptr, T* C_ptr, size_t M, size_t N, size_t K) {
		for (size_t m = 0; m < M; ++m) {
			for (size_t n = 0; n < N; ++n) {
				T entry = T();
				for (size_t k = 0; k < K; ++k) {
					entry += A_ptr[k + m * K] * B_ptr[k * N + n];
				}
				C_ptr[m * N + n] = entry;
			}
		}
	}

public:

	// ==================== Constructor ======================
	/*
		Constructs a N-dimensional array (tensor) using the specified shape.

		Params: 
			shape_input: std::vector<size_t>{... args},  
	
	*/
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

	void set_data(const std::vector<T>& input_data) {
		if (input_data.size() != data.size()) {
			throw std::invalid_argument(std::format("The size of the data does not match the internal data size, input size should be {}", data.size()));
		}
		this->data = input_data;
	}

	void set_data(std::vector<T>&& input_data) {
		if (input_data.size() != data.size()) {
			throw std::invalid_argument(std::format("The size of the data does not match the internal data size, input size should be {}", data.size()));
		}
		this->data = std::move(input_data);
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
	/*
		Performs elementwise tensor additions.
		
		Params:
			other: The second ndarray we are adding with.
	*/
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

	/*
		Checks if two ndarrays are equivalent.

		Params:
			other: The second ndarray;
	*/
	bool operator==(const NDArray<T>& other) const {
		return (shape == other.shape) && (data == other.data);
	}

	/*
		Performs element-wise scalar multiplication;

		Params:
			scalar: the value the tensor is multiplying by.
	*/
	NDArray<T> operator*(T scalar) const {
		NDArray<T> result(shape);
		for (size_t i = 0; i < data.size(); ++i) {
			result[i] = scalar * data[i];
		}
		return result;
	}

	// ----------------- Transpose -----------------

	/*
		Returns an ndarray with the specified dimensions swapped.
		Parameters: 
			dim1: the first dimension to swap with
			dim2: the second dimension to swap with

	*/
	NDArray<T> transpose(const NDArray& other, size_t dim1, size_t dim2) const {
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

		Params: 
			other: The second tensor (ndarray)
	*/
	NDArray<T> batched_matmul(const NDArray& other) const {
		if (other.shape.size() != shape.size()) {
			throw std::invalid_argument("The dimensions of the ndarrays do not match!");
		}
		if (other.shape.size() == 2 && shape.size() == 2) {
			// Handle 2D case. 
			return this->matmul(other);
		}

		if (!std::equal(shape.begin(), shape.end() - 2, other.shape.begin())) {
			throw std::invalid_argument("The batches for each ndarray must be the same!");
		}

		if (shape[shape.size() - 1] != other.shape[other.shape.size() - 2]) {
			throw std::invalid_argument("The shape of the two ndarrays do not match!");
		}

		// TODO: implement the batched matrix multiplication;
		size_t M = shape[shape.size() - 2];
		size_t N = other.shape[other.shape.size() - 1];
		size_t K = shape[shape.size() - 1];

		size_t size_A = M * K;
		size_t size_B = K * N;
		size_t size_C = M * N;

		// Parepare the result NDArray:
		std::vector<size_t> res_shape(shape.begin(), shape.end() - 2);
		res_shape.push_back(M);
		res_shape.push_back(N);
		NDArray<T> res(res_shape);

		// Get the pointers
		const T* ptr_A = data.data();
		const T* ptr_B = other.data.data();
		T* ptr_C = res.data.data();

		// Compute the size of the batch
		size_t total_elements = data.size();
		size_t batch_count = total_elements / size_A;

		for (size_t i = 0; i < batch_count; ++i) {
			_matmul(ptr_A, ptr_B, ptr_C, M, N, K);

			ptr_A += size_A;
			ptr_B += size_B;
			ptr_C += size_C;
		}

		return res;
	}


	/*
		Computes the product of 2 matrices. Standard matrix multiplcation. (Recall that matrices are by definition 2D).

		Params:
			other: The second matrix.
	*/
	NDArray<T> matmul(const NDArray& other) const {
		if (shape.size() != 2 || other.shape.size() != 2) {
			throw std::invalid_argument("Each ndarray must be a matrix (2D NDArray)!");
		}
		if (shape[1] != other.shape[0]) {
			throw std::invalid_argument("The number of columns in your first matrix does not align with the number of rows in your second matrix!");
		}

		size_t M = shape[0];
		size_t N = other.shape[1];
		size_t K = shape[1];

		NDArray<T> result({M, N});
		
		// Recall taking the .data() property of a vector yields the pointer that points to the first value of the vector.
		_matmul(data.data(), other.data.data(), result.data.data(), M, N, K);

		return result;
	}


	/*
		Performs matrix multiplication, strictly restricsts the dimension of each array to be 2D.
		Legacy Implementation: very slow if we were to reuse this function for the batched matrix multiplication. This is a performance killer for math library, so we kinda need to pivot away from this approach.
	
		Params:
			other: The second matrix.
	*/
	NDArray<T> matmul_legacy(const NDArray& other) const {
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

	/*
		Prints the tensor in falttened form.
	*/
	void print_data() const {
		std::cout << "[ ";
		for (size_t i = 0; i < data.size(); ++i) {
			std::cout << data[i] << " ";
		}
		std::cout << "]" << std::endl;
	}

	/*
		Prints the shape of the tensor.
	*/
	void print_shape() const{
		std::cout << "( ";
		for (size_t i = 0; i < shape.size(); i++) {
			std::cout << shape[i] << ", ";
		}
		std::cout << ")" << std::endl;
	}


	// ================== Getter Functions ===========================
	/*
		returns the reference to the data attribute
	*/
	std::vector<T>& get_data() {
		return data;
	}

	/*
		returns the reference the shape
	*/
	std::vector<size_t>& get_shape() {
		return shape;
	}

	/* returns the strides of the ndarray */
	std::vector<size_t>& get_strides() {
		return strides;
	}
};
