#include "stdafx.h"
#include "Layer.h"
#include "MatrixOps.h"

InputLayer::InputLayer()
{
	trains = new bool*[784];
	alphas = new double*[7840];

	srand(time(NULL));

	//generate alphas randomly
	for (int i = 0; i < 7840; ++i)
	{
		alphas[i] = new double[TYI];

		for (int j = 0; j < TYI; ++j)
			alphas[i][j] = (rand() % 10 + 1) / 50.0; //0.02 to 0.2
	}
}

void InputLayer::AddTrain(bool * train)
{
	trains[index++] = train;
}

void InputLayer::ResetTrains()
{
	for (int i = 0; i < 784; ++i)
		delete trains[i];

	delete trains;

	trains = new bool*[784];
}

double* InputLayer::ApplyAlphas() const
{
	double** result = new double*[CLASSES*NEURONS_IN];
	short j = 0; //index of trains, which increases once evry 10 i
	for (short i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		result[i] = MatrixOps::Conv(trains[j], alphas[i]);
		if (++j == CLASSES)
			j = 0;
	}

	return MatrixOps::SumColumns(result);
}



void InputLayer::UpdateAlphas(double** errors)
{
	short j = 0; //index of trains, which increases once evry 10 i
	for (short i = 0; i < 7840; ++i)
	{
		for (short t = 0; t < T; ++t)
		{
			alphas[i][t] += LEARNING_RATE * (errors[i][j] * trains[i][j]);
		}
		if (++j == 10)
			j = 0;
	}
}

InputLayer::~InputLayer()
{
	for (int i = 0; i < 784; ++i)
		delete trains[i];

	for (int i = 0; i < 7840; ++i)
		delete alphas[i];

	delete trains;
	delete alphas;
}

OutputLayer::OutputLayer()
{
	betas = new double*[10];
	u = new double*[10];
	y = new bool*[10];
	gammas = new double[10];

	srand(time(NULL));

	//generate betas randomly
	for (int i = 0; i < 10; ++i)
	{
		betas[i] = new double[TYO];

		for (int j = 0; j < TYO; ++j)
			betas[i][j] = (rand() % 10 + 1) / 50.0; //0.02 to 0.2
	}
}

OutputLayer::~OutputLayer()
{
	for (short i = 0; i < 10; ++i) {
		delete betas[i];
		//delete u[i];
		//delete y[i]; TODO rollback
	}

	delete betas;
	delete gammas;
	delete u;
	delete y;
}

void OutputLayer::Reset()
{
	/*for (short i = 0; i < CLASSES; ++i) {
		delete u[i];
		delete y[i]; TODO rollback
	}*/

	delete u;
	delete y;

	u = new double*[CLASSES];
	y = new bool*[CLASSES];
}

void OutputLayer::ComputeOutput(double ** synapsesOut)
{
	
}

char OutputLayer::ComputeWinner() const
{
	char bestIndex = 0;
	char bestSpikes = 0;

	for (short i = 0; i < 10; ++i) 
	{
		char currentSpikes = 0;
		for (short t = 0; t < T; ++t)
			currentSpikes += y[i][t];

		if (currentSpikes > bestSpikes) 
		{
			bestSpikes = currentSpikes;
			bestIndex = i;
		}
	}

	return bestIndex;
}
