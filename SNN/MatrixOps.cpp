#include "stdafx.h"
#include "MatrixOps.h"

using std::array;

double* MatrixOps::Conv(const bool * f, const double * g)
{
	int const n = T + TYI - 1;
	int const out_size = T - 1;
	double* out = new double[out_size];
	for (int z = 0; z < out_size; ++z)
		out[z] = 0;
	for (auto i(0); i < out_size; ++i) {
		int const jmn = (i >= TYI - 1) ? i - (TYI - 1) : 0;
		int const jmx = (i <  out_size) ? i : out_size;
		for (auto j(jmn); j <= jmx; ++j) {
			out[i] += (f[j] * g[i - j]);
		}
	}
	return out;
}

array<double, T-1> MatrixOps::Conv(array<bool, T> const & f, array<double, TYI> const & g)
{
	int const out_size = T - 1;
	array<double, T - 1> out{};
	for (auto i(0); i < out_size; ++i) {
		int const jmn = (i >= TYI - 1) ? i - (TYI - 1) : 0;
		int const jmx = (i <  out_size) ? i : out_size;
		for (auto j(jmn); j <= jmx; ++j) {
			out[i] += (f[j] * g[i - j]);
		}
	}
	return out;
}

double* MatrixOps::Multiply(const bool * f, const double * g, const short len)
{
	double* res = new double[len];
	for (short i = 0; i < len; ++i)
		res[i] = f[i] * g[i];

	return res;
}

/*array<double> MatrixOps::SumColumns(array<array<double>>& vect)
{
	array<double> tot = array<double>(vect[0].size());

	for (int i = 0; i < vect[0].size(); ++i)
	{
		tot[i] = 0;
		for (int j = 0; j < vect.size(); ++j)
		{
			tot[i] += vect[j][i];
		}
	}

	return tot;
}*/

/*
 Sum the columns where the row's reminder is relevant
*/
array<double, T-1> MatrixOps::SumColumnsMod(array<array<double, T-1>, CLASSES*NEURONS_IN>& vect, const short cl)
{
	array<double, T-1> tot = array<double, T-1>();

	for (int i = 0; i < vect[0].size(); ++i)
	{
		tot[i] = 0;
		for (int j = cl; j < vect.size(); j+=CLASSES)
		{
			tot[i] += vect[j][i];
		}	
	}

	return tot;
}
