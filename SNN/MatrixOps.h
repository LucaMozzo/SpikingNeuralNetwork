#pragma once
#include "stdafx.h"

using std::array;

class MatrixOps
{
public:
	//Performs a reduced convolution to aggregate the trains and alpha
	static double* Conv(const bool* f, const double* g);
	static array<double, T-1> Conv(array<bool,T> const &f, array<double,TYI> const &g);
	static double* Multiply(const bool* f, const double* g, const short len);
	//static array<double> SumColumns(array<array<double>>& vect);
	static array<double, T-1> SumColumnsMod(array<array<double, T-1>, CLASSES*NEURONS_IN>& vect, const short cl);

	static array<double, T - 1> SumColumnsMod2(array<array<double, T - 1>, HIDDEN_NEURONS*CLASSES>& vect, const short cl);

	template<std::size_t SIZE>
	static double Sum(const array<double, SIZE>& vect);

	template<std::size_t ROWS, std::size_t COLS>
	static array<double, COLS> SumColumns(array<array<double, COLS>, ROWS>& vect);
};

template<std::size_t SIZE>
inline double MatrixOps::Sum(const array<double, SIZE>& vect)
{
	double tot = 0;

	for (int i = 0; i < vect.size(); ++i)
		tot += vect[i];

	return tot;
}

template <std::size_t ROWS, std::size_t COLS>
array<double, COLS> MatrixOps::SumColumns(array<array<double, COLS>, ROWS>& vect)
{
	array<double, COLS> tot = array<double, COLS>();

	for (int i = 0; i < vect[0].size(); ++i)
	{
		tot[i] = 0;
		for (int j = 0; j < vect.size(); ++j)
		{
			tot[i] += vect[j][i];
		}
	}

	return tot;
}
