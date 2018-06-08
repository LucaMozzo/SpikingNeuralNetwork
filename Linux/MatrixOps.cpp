#include "MatrixOps.h"

using std::array;


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

array<double, T-1> MatrixOps::SumColumns(array<array<double, T-1>, CLASSES*NEURONS_IN>& vect, const short cl)
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

