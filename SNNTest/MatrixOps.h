#pragma once
#include "stdafx.h"

using std::vector;

class MatrixOps
{
public:
	//Performs a reduced convolution to aggregate the trains and alpha
	static double* Conv(const bool* f, const double* g);
	static vector<double> Conv(vector<bool> const &f, vector<double> const &g);
	static double* Multiply(const bool* f, const double* g, const short len);
	static double* SumColumns(const double** vect, const short rows = CLASSES * NEURONS_IN, const short columns = T);
	static vector<double> SumColumnsMod(vector<vector<double>>& vect, const short cl);
	static double Sum(const double* vect, const short size);
	static double MatrixOps::Sum(const vector<double>& vect);
};