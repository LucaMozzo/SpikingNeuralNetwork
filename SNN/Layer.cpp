#include "stdafx.h"
#include "Layer.h"

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

InputLayer::~InputLayer()
{
	for (int i = 0; i < 784; ++i)
		delete trains[i];

	for (int i = 0; i < 7840; ++i)
		delete alphas[i];

	delete trains;
	delete alphas;
}
