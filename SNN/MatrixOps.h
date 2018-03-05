#pragma once
#include "stdafx.h"

class MatrixOps
{
public:
	//Performs a reduced convolution to aggregate the trains and alpha
	static double* Conv(const bool* f, const double* g);
	static double* Multiply(const bool* f, const double* g, const short len);
	static double* SumColumns(const double** vect, const short rows = CLASSES * NEURONS_IN, const short columns = T);
};