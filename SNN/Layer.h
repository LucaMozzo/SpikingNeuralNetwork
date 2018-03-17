#pragma once
#include "stdafx.h"
#include <time.h>
#include "MatrixOps.h"

using std::array;

class InputLayer
{
protected:

	array<array<bool, T>, NEURONS_IN> trains;
	short index = 0;

public:

	array<array<double, TYI>, CLASSES*NEURONS_IN> alphas;

	InputLayer();
	void AddTrain(array<bool, T>& train);
	void ResetTrains();
	array<array<double, T-1>, CLASSES*NEURONS_IN> ApplyAlphas() const;
	void UpdateAlphas(array<array<double, T>, CLASSES>& errors);
};

class OutputLayer
{
protected:

	array<array<double, T>, CLASSES> u;
	array<array<bool, T>, CLASSES> y;

public:

	array<array<double, TYO>, CLASSES> betas;
	array<double, CLASSES> gammas;

	OutputLayer();
	void Reset();
	void ComputeOutput(array<array<double, T-1>, CLASSES*NEURONS_IN>& synapsesOut);
	array<array<double, T>, CLASSES> ComputeErrors(unsigned char label) const;
	char ComputeWinner() const;
	void UpdateBetas(array<array<double, T>, CLASSES>& errors);
	void UpdateGammas(array<array<double, T>, CLASSES>& errors);
};

/*class HiddenLayer : OutputLayer
{
public:
	array<array<double, TYI>, 20> alphas;

	
};*/