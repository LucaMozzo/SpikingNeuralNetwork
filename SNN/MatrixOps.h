#pragma once
#include "stdafx.h"

using std::array;

class MatrixOps
{
public:
	//Performs a reduced convolution to aggregate the trains and alpha
	[[deprecated]]
	static double* Conv(const bool* f, const double* g);
	static array<double, T-1> Conv(array<bool,T> const &f, array<double,TYI> const &g);
	[[deprecated]]
	static double* Multiply(const bool* f, const double* g, const short len);
	//static array<double> SumColumns(array<array<double>>& vect);
	static array<double, T-1> SumColumnsMod(array<array<double, T-1>, CLASSES*NEURONS_IN>& vect, const short cl);
	
	template<std::size_t ROWS, std::size_t COLS>
	static array<double, ROWS> Dot(array<array<double, COLS>, ROWS>& basis, array<double, COLS>& weight);

	template<std::size_t SIZE>
	static double Sum(const array<double, SIZE>& vect);

	template<std::size_t ROWS, std::size_t COLS>
	static array<array<double, ROWS>, COLS> Transpose(array<array<double, COLS>, ROWS>& input);
};

// implementation of the template methods 
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
