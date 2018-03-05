#include "stdafx.h"
#include "MatrixOps.h"

double * MatrixOps::Conv(const bool * f, const double * g)
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
