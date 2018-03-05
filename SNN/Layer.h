#pragma once
#include "stdafx.h"
#include <time.h>

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
	double** ApplyAlphas();
	void UpdateAlphas(double* errors);
};

class OutputLayer
{
protected:

	double** betas;
	double* gammas;
	double** potentials;
	bool** y;

public:

	OutputLayer();
	~OutputLayer();
	bool** ComputeOutput();
	double** ComputeErrors();
	char ComputeWinner();
	void UpdateBetas(double* errors);
	void UpdateGammas(double* errors);
};