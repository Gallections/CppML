// CppML.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include "NDArray.hpp"
#include <vector>

int main()
{
    std::cout << "This is the start of the CppML library" << std::endl;
    NDArray<int> arrayOne(std::vector<size_t>{1, 2, 3});
    arrayOne.print_data();
    arrayOne.print_shape();

    NDArray<int> m1(std::vector<size_t>{2, 2});
    NDArray<int> m2(std::vector<size_t>{2, 3});

    m1(std::vector<size_t>{ 0, 0 }) = 1;
    m1(std::vector<size_t>{ 0, 1 }) = 2;
    m1(std::vector<size_t>{ 1, 0 }) = 3;
    m1(std::vector<size_t>{ 1, 1 }) = 4;

    m2(std::vector<size_t>{ 0, 0 }) = 1;
    m2(std::vector<size_t>{ 0, 1 }) = 2;
    m2(std::vector<size_t>{ 0, 2 }) = 3;
    m2(std::vector<size_t>{ 1, 0 }) = 4;
    m2(std::vector<size_t>{ 1, 1 }) = 5;
    m2(std::vector<size_t>{ 1, 2 }) = 6;

    m1.print_data();
    m2.print_data();
    m1.matmul2D(m2).print_data();


}
