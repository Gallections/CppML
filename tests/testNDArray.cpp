#include <gtest/gtest.h>
#include "NDArray.hpp"
#include <vector>
#include <stdexcept>

TEST(SanityCheck, BasicMath) {
	EXPECT_EQ(1 + 1, 2);
}


// ============== NDArrayConstruction =================
TEST(NDArrayConstruction, Constructing2DArray) {
	std::vector<size_t> input_shape({ 3, 3 });
	NDArray<int> threeByThree(std::move(input_shape));

	std::vector<size_t>& shape = threeByThree.get_shape();
	std::vector<int>& data = threeByThree.get_data();
	std::vector<size_t>& strides = threeByThree.get_strides();
	
	EXPECT_EQ(data.size(), 9);
	EXPECT_EQ(shape.size(), 2);
	EXPECT_EQ(strides.size(), 2);

	EXPECT_EQ(shape, input_shape);
	EXPECT_EQ(strides, std::vector<size_t>({3, 1}));
}

TEST(NDArrayConstruction, SetDataSuccess) {
	std::vector<size_t> input_shape({ 2, 2 });
	NDArray<int> twoByTwo(std::move(input_shape));

	EXPECT_NO_THROW(twoByTwo.set_data({ 1, 2, 3, 4}));

	std::vector<int> data = twoByTwo.get_data();
	EXPECT_EQ(data, std::vector<int>({1, 2, 3, 4}));
}

TEST(NDArrayConstruction, SetDataSizeUnMatch) {
	std::vector<size_t> input_shape({ 2, 2 });
	NDArray<int> threeByThree(std::move(input_shape));

	EXPECT_THROW(threeByThree.set_data({ 1, 2, 3 }), std::invalid_argument);
}

TEST(NDArrayConstruction, ConstructingNDArray) {
	std::vector<size_t> input_shape({3, 4, 2});
	NDArray<int> ndarray(std::move(input_shape));

	std::vector<size_t>& shape = ndarray.get_shape();
	std::vector<int>& data = ndarray.get_data();
	std::vector<size_t>& strides = ndarray.get_strides();

	EXPECT_EQ(data.size(), 24);
	EXPECT_EQ(shape.size(), 3);
	EXPECT_EQ(strides.size(), 3);

	EXPECT_EQ(shape, input_shape);
	EXPECT_EQ(strides, std::vector<size_t>({8, 2, 1}));
}


// ============== NDArray internal properties ==================
TEST(NDArrayInternalProperties, twoDTranspose) {
	NDArray<int> m1({ 1, 2 });
	NDArray<int> res = m1.transpose(0, 1);
	std::vector<size_t> res_shape({2, 1});
	std::vector<size_t> res_strides({1, 1});

	EXPECT_EQ(res.get_shape(), res_shape);
	EXPECT_EQ(res.get_strides(), res_strides);
}

TEST(NDArrayInternalProperties, NDTranspose) {
	NDArray<int> m1({ 3, 5, 7, 2});
	NDArray<int> res = m1.transpose(0, 3);
	std::vector<size_t> res_shape({ 2, 5, 7, 3 });
	std::vector<size_t> res_strides({ 105, 21, 3, 1 });

	EXPECT_EQ(res.get_shape(), res_shape);
	EXPECT_EQ(res.get_strides(), res_strides);
}

TEST(NDArrayInternalProperties, squaring) {
	NDArray<int> m1({ 2, 2 });
	NDArray<float> m2({2, 1});

	m1.set_data({2, 1, 3, 4});
	m2.set_data({1.23f, 2.004f});

	std::vector<int> expected_m1({4, 1, 9, 16});
	std::vector<float> expected_m2({1.5129f, 4.016016f});

	NDArray<int> m1_sq = m1.square();
	NDArray<float> m2_sq = m2.square();

	EXPECT_EQ(expected_m1, m1_sq.get_data());

	ASSERT_EQ(expected_m2.size(), m2_sq.get_data().size());
	for (size_t i = 0; i < expected_m2.size(); ++i) {
		EXPECT_FLOAT_EQ(expected_m2[i], m2_sq.get_data()[i]);
	}
}

TEST(NDArrayInternalProperties, summing) {
	NDArray<int> m1({ 2, 2, 2 });
	m1.set_data({1, 2, 3, 4, 5, 6, 7, 8});
	EXPECT_EQ(m1.sum(), 36);
}

// ============== 2DArrayMultiplication ================
TEST(TwoDArrayMultiplication, SimpleSuccess1) {
	NDArray<int> m1({1, 2});
	m1.set_data({2, 2});
	NDArray<int> m2({ 2, 3 });
	m2.set_data({ 1, 2, 3, 4, 5, 6 });

	EXPECT_NO_THROW(m1.matmul(m2));
	NDArray<int> res = m1.matmul(m2);
	EXPECT_EQ(res.get_data(), std::vector<int>({ 10, 14, 18 }));
	EXPECT_EQ(res.get_shape(), std::vector<size_t>({ 1, 3 }));
}

TEST(TwoDArrayMultiplication, SimpleSuccess2) {
	NDArray<float> m1({ 1, 2 });
	m1.set_data({ 2.2f, 2.3f });
	NDArray<float> m2({ 2, 3 });
	m2.set_data({ 1.1f, 2.5f, 3.9f, 4.2f, 5.6f, 6.2f });

	EXPECT_NO_THROW(m1.matmul(m2));
	NDArray<float> res = m1.matmul(m2);
	std::vector<float> expected({ 12.08f, 18.38f, 22.84f });
	std::vector<float> data = res.get_data();

	ASSERT_EQ(data.size(), expected.size());
	for (size_t i = 0; i < data.size(); ++i) {
		EXPECT_FLOAT_EQ(expected[i], data[i]);
	}

	ASSERT_EQ(res.get_shape(), std::vector<size_t>({ 1, 3 }));
}

TEST(TwoDArrayMultiplication, ComplexSuccess) {
	NDArray<int> m1({ 3, 4 });
	m1.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
	NDArray<int> m2({ 4, 5 });
	m2.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });

	EXPECT_NO_THROW(m1.matmul(m2));
	NDArray<int> res = m1.matmul(m2);
	EXPECT_EQ(res.get_data(), std::vector<int>({ 110, 120, 130, 140, 150, 246, 272, 298, 324, 350, 382, 424, 466, 508, 550 }));
	EXPECT_EQ(res.get_shape(), std::vector<size_t>({ 3, 5 }));
}

TEST(TwoDArrayMultiplication, ShapeUnmatch) {
	NDArray<int> m1({ 1, 2 });
	m1.set_data({ 2, 2 });
	NDArray<int> m2({ 3, 3 });
	m2.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9});

	EXPECT_THROW(m1.matmul(m2), std::invalid_argument);
}

TEST(TwoDArrayMultiplication, SizeUnMatch) {
	NDArray<int> m1({ 1, 2 });
	m1.set_data({ 2, 2 });
	NDArray<int> m2({ 2, 1, 1 });
	m2.set_data({ 1, 2 });

	EXPECT_THROW(m1.matmul(m2), std::invalid_argument);
}


// ============== Batched Matrix Multiplication ======================
TEST(BatchedMatrixMultiplication, TwoDCase) {
	NDArray<int> m1({ 1, 2 });
	m1.set_data({ 2, 2 });
	NDArray<int> m2({ 2, 3 });
	m2.set_data({ 1, 2, 3, 4, 5, 6 });

	EXPECT_NO_THROW(m1.batched_matmul(m2));
	NDArray<int> res = m1.batched_matmul(m2);
	EXPECT_EQ(res.get_data(), std::vector<int>({ 10, 14, 18 }));
	EXPECT_EQ(res.get_shape(), std::vector<size_t>({ 1, 3 }));
}

TEST(BatchedMatrixMultiplication, TwoDCaseComplex) {
	NDArray<int> m1({ 3, 4 });
	m1.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
	NDArray<int> m2({ 4, 5 });
	m2.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 });

	EXPECT_NO_THROW(m1.batched_matmul(m2));
	NDArray<int> res = m1.batched_matmul(m2);
	EXPECT_EQ(res.get_data(), std::vector<int>({ 110, 120, 130, 140, 150, 246, 272, 298, 324, 350, 382, 424, 466, 508, 550 }));
	EXPECT_EQ(res.get_shape(), std::vector<size_t>({ 3, 5 }));
}

TEST(BatchedMatrixMultiplication, ShapeUnmatch) {
	NDArray<int> m1({ 1, 2 });
	m1.set_data({ 2, 2 });
	NDArray<int> m2({ 3, 3 });
	m2.set_data({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

	EXPECT_THROW(m1.batched_matmul(m2), std::invalid_argument);
}

TEST(BatchedMatrixMultiplication, SizeUnMatch) {
	NDArray<int> m1({ 1, 2 });
	m1.set_data({ 2, 2 });
	NDArray<int> m2({ 2, 1, 1 });
	m2.set_data({ 1, 2 });

	EXPECT_THROW(m1.batched_matmul(m2), std::invalid_argument);
}

TEST(BatchedMatrixMultiplication, NDCase1) {
	NDArray<int> m1({ 3, 2, 1, 2 });
	m1.set_data({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2});
	NDArray<int> m2({ 3, 2, 2, 3 });
	m2.set_data({ 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
		1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
		1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 });

	std::vector<int> expected({2, 2, 2, 4, 4, 4, 2, 2, 2, 8, 8, 8, 4, 4, 4, 8, 8, 8});
	NDArray<int> res = m1.batched_matmul(m2);
	ASSERT_EQ(expected.size(), res.get_data().size());

	EXPECT_EQ(res.get_shape(), std::vector<size_t>({3, 2, 1, 3}));
	EXPECT_EQ(res.get_data(), expected);
}

TEST(BatchedMatrixMultiplication, NDCase2) {
	NDArray<int> m1({ 2, 1, 2 });
	m1.set_data({ 1, 2, 3, 4 });
	NDArray<int> m2({ 2, 2, 1 });
	m2.set_data({ 4, 3, 2, 1 });

	std::vector<int> expected({ 10, 10 });
	NDArray<int> res = m1.batched_matmul(m2);
	ASSERT_EQ(expected.size(), res.get_data().size());

	EXPECT_EQ(res.get_shape(), std::vector<size_t>({ 2, 1, 1 }));
	EXPECT_EQ(res.get_data(), expected);
}

TEST(BatchedMatrixMultiplication, ShapeUnmatchCase1) {
	NDArray<int> m1({ 3, 2, 1, 2 });
	NDArray<int> m2({ 3, 2, 3, 3 });
	EXPECT_THROW(m1.batched_matmul(m2), std::invalid_argument);
}

TEST(BatchedMatrixMultiplication, ShapeUnmatchCase2) {
	NDArray<int> m1({ 3, 2, 1, 2 });
	NDArray<int> m2({ 4, 5, 2, 2, 3 });
	EXPECT_THROW(m1.batched_matmul(m2), std::invalid_argument);
}

TEST(BatchedMatrixMultiplication, ShapeUnmatchCase3) {
	NDArray<int> m1({ 3, 2, 1, 2 });
	NDArray<int> m2({ 3, 2, 3 });
	EXPECT_THROW(m1.batched_matmul(m2), std::invalid_argument);
}
