#pragma once
#include <iostream>
#include <array>
#include "Constants.h"
using std::array;

/**
This class contains custom matrix operations
*/
class MatrixOps
{
public:
	/**
	Performs a partial convolution to aggregate the trains and alpha
	@param f The spike train
	@param g The weights vector
	@returns The result of the convolution
	*/
	static array<double, T-1> Conv(array<bool,T> const &f, array<double,TYI> const &g);
	/**
	Sum the matrix along the columns, but only every cl rows
	@param vect The matrix
	@param cl The frequency of rows to be summed, i.e. cl=2 half of the rows are summed
	@return The sum
	*/
	static array<double, T-1> SumColumns(array<array<double, T-1>, CLASSES*NEURONS_IN>& vect, const short cl);
	/**
	Multiplies an array by a constant
	@param SIZE The length of the array
	@param c The constant
	@param vect The array
	*/
	template<std::size_t SIZE>
	static void Multiply(double c, array<double, SIZE>& vect);
	/**
	Performs the dot product
	@param ROWS The number of rows
	@param COLS The number of columns
	@param basis The basis matrix
	@param weight The weight vector
	@returns the result of the operation
	*/
	template<std::size_t ROWS, std::size_t COLS>
	static array<double, ROWS> Dot(array<array<double, COLS>, ROWS>& basis, array<double, COLS>& weight);
	/**
	Sums all the values in a vector
	@param SIZE The length of the array
	@param ROWS The number of rows
	@param COLS The number of columns
	@param vect The vector
	@return The sum of the elements
	*/
	template<std::size_t SIZE>
	static double Sum(const array<double, SIZE>& vect);
	/**
	Performs a matrix transpose
	@param ROWS The number of rows
	@param COLS The number of columns
	@param input The input matrix
	@return The transposed matrix
	*/
	template<std::size_t ROWS, std::size_t COLS>
	static array<array<double, ROWS>, COLS> Transpose(array<array<double, COLS>, ROWS>& input);
	/**
	Performs element-wise sum of the elements in the arrays
	@param SIZE The length of the arrays
	@param a Array 1
	@param b Array 2
	@return The element-wise sum of the arrays
	*/
	template<std::size_t SIZE>
	static array<double, SIZE> SumArrays(array<double, SIZE> a, array<double, SIZE> b);
};
// implementation of the template methods 
template<std::size_t SIZE>
inline void MatrixOps::Multiply(double c, array<double, SIZE>& vect)
{
	for (short i = 0; i < SIZE; ++i)
		vect[i] = vect[i] * c;
}

template<std::size_t ROWS, std::size_t COLS>
inline array<double, ROWS> MatrixOps::Dot(array<array<double, COLS>, ROWS>& basis, array<double, COLS>& weight)
{
	auto res = array<double, ROWS>();
	
	short index = 0;

	for(short r = 0; r < ROWS; ++r)
	{
		double accumulator = 0;
		for(short c = 0; c < COLS; ++c)
		{
			accumulator += basis[r][c] * weight[c];
		}
		res[index++] = accumulator;
	}

	return res;
}

template<std::size_t SIZE>
inline double MatrixOps::Sum(const array<double, SIZE>& vect)
{
	double tot = 0;

	for (int i = 0; i < vect.size(); ++i)
		tot += vect[i];

	return tot;
}

template<std::size_t ROWS, std::size_t COLS>
inline array<array<double, ROWS>, COLS> MatrixOps::Transpose(array<array<double, COLS>, ROWS>& input)
{
	auto res  = array<array<double, ROWS>, COLS>();

	for (short r = 0; r < ROWS; ++r)
	{
		for (short c = 0; c < COLS; ++c)
		{
			res[c][r] = input[r][c];
		}
	}

	return res;
}

template<std::size_t SIZE>
inline array<double, SIZE> MatrixOps::SumArrays(array<double, SIZE> a, array<double, SIZE> b)
{
	auto res = array<double, SIZE>();

	for (short i = 0; i < SIZE; ++i)
		res[i] = a[i] + b[i];

	return res;
}
