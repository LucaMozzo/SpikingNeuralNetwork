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

vector<double> MatrixOps::Conv(vector<bool> const & f, vector<double> const & g)
{
	int const nf = f.size();
	int const ng = g.size();
	int const n = nf + ng - 1;
	int const out_size = nf - 1;
	vector<double> out(out_size, 0);
	for (auto i(0); i < out_size; ++i) {
		int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
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

double* MatrixOps::SumColumns(const double** vect, const short rows, const short columns)
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

/*
 Sum the columns where the row's reminder is relevant
*/
vector<double> MatrixOps::SumColumnsMod(vector<vector<double>>& vect, const short cl)
{
	vector<double> tot = vector<double>(vect[0].size());

	for (int i = 0; i < vect[0].size(); ++i)
	{
		tot[i] = 0;
		for (int j = cl; j < vect.size(); j+=CLASSES)
		{
			tot[i] += vect[j][i];
		}	}

	return tot;
}

 double MatrixOps::Sum(const double * vect, const short size)
 {
	 double tot = 0;

	 for (int i = 0; i < size; ++i)
		 tot += vect[i];

	 return tot;
 }

 double MatrixOps::Sum(const vector<double>& vect)
 {
	 double tot = 0;

	 for (int i = 0; i < vect.size(); ++i)
		 tot += vect[i];

	 return tot;
 }
