#pragma once
#include "stdafx.h"
#include <time.h>
#include "MatrixOps.h"

using std::vector;

class InputLayer
{
protected:

	vector<vector<bool>> trains;
	vector<vector<double>> alphas;
	short index = 0;

public:

	InputLayer();
	void AddTrain(vector<bool>& train);
	void ResetTrains();
	vector<vector<double>> ApplyAlphas() const;
	void UpdateAlphas(double** errors);
};

class OutputLayer
{
protected:

	vector<vector<double>> betas;
	vector<double> gammas;
	vector<vector<double>> u;
	vector<vector<bool>> y;

public:

	OutputLayer();
	void Reset();
	void ComputeOutput(vector<vector<double>>& synapsesOut);
	//double** ComputeErrors() const;
	char ComputeWinner() const;
	//void UpdateBetas(double* errors);
	//void UpdateGammas(double* errors);
};