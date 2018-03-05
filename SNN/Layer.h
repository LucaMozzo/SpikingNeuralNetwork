#pragma once
#include "stdafx.h"
#include <time.h>
#include "MatrixOps.h"

class InputLayer
{
protected:

	bool** trains;
	double** alphas;
	short index = 0;

public:

	InputLayer();
	~InputLayer();
	void AddTrain(bool* train);
	void ResetTrains();
	double* ApplyAlphas() const;
	void UpdateAlphas(double** errors);
};

class OutputLayer
{
protected:

	double** betas;
	double* gammas;
	double** u;
	bool** y;

public:

	OutputLayer();
	~OutputLayer();
	void Reset();
	void ComputeOutput(double** synapsesOut);
	double** ComputeErrors() const;
	char ComputeWinner() const;
	void UpdateBetas(double* errors);
	void UpdateGammas(double* errors);
};