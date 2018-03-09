#pragma once
#include "stdafx.h"
#include <time.h>
#include "MatrixOps.h"

using std::vector;

class InputLayer
{
protected:

	vector<vector<bool>> trains;
	short index = 0;

public:

	vector<vector<double>> alphas;

	InputLayer();
	void AddTrain(vector<bool>& train);
	void ResetTrains();
	vector<vector<double>> ApplyAlphas() const;
	void UpdateAlphas(vector<vector<double>>& errors);
};

class OutputLayer
{
protected:

	vector<vector<double>> u;
	vector<vector<bool>> y;

public:

	vector<vector<double>> betas;
	vector<double> gammas;

	OutputLayer();
	void Reset();
	void ComputeOutput(vector<vector<double>>& synapsesOut);
	vector<vector<double>> ComputeErrors(unsigned char label) const;
	char ComputeWinner() const;
	void UpdateBetas(vector<vector<double>> errors);
	void UpdateGammas(vector<vector<double>> errors);
};