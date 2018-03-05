#include "stdafx.h"
#include "MatrixOps.h"

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

double* MatrixOps::Multiply(const bool * f, const double * g, const short len)
{
	double* res = new double[len];
	for (short i = 0; i < len; ++i)
		res[i] = f[i] * g[i];

	return res;
}

double* MatrixOps::SumColumns(const double** vect, const short rows=CLASSES* NEURONS_IN, const short columns=T)
{
	double* tot = new double[columns];

	for (int i = 0; i < columns; ++i)
	{
		tot[i] = 0;
		for (int j = 0; j < rows; ++j)
		{
			tot[i] += vect[j][i];
		}
	}

	return tot;
}