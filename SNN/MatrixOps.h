#pragma once
#include "stdafx.h"

class MatrixOps
{
public:
	//Performs a reduced convolution to aggregate the trains and alpha
	static double* Conv(const bool* f, const double* g);
};