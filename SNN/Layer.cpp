#include "stdafx.h"
#include "Layer.h"

InputLayer::InputLayer()
{
	trains = vector<vector<bool>>(NEURONS_IN);
	alphas = vector<vector<double>>(NEURONS_IN*CLASSES);

	srand(time(NULL));

	//generate alphas randomly
	for (int i = 0; i < 7840; ++i)
	{
		alphas[i] = vector<double>(TYI);

		for (int j = 0; j < TYI; ++j)
			alphas[i][j] = (rand() % 10 + 1) / 50.0; //0.02 to 0.2
	}
}

void InputLayer::AddTrain(vector<bool>& train)
{
	trains[index++] = train;
}

void InputLayer::ResetTrains()
{
	index = 0;
	trains = vector<vector<bool>>(NEURONS_IN);
}

vector<vector<double>> InputLayer::ApplyAlphas() const
{
	vector<vector<double>> result = vector<vector<double>>(CLASSES*NEURONS_IN);
	short j = 0; //index of trains, which increases once every 10 i
	for (short i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		result[i] = MatrixOps::Conv(trains[j], alphas[i]);
		if (++j == CLASSES)
			j = 0;
	}

	return result;
}

void InputLayer::UpdateAlphas(vector<vector<double>> errors)
{
	short j = 0; //index of trains, which increases once every 10 i
	for (short i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		for (short t = 0; t < T; ++t)
		{
			alphas[i][t] += LEARNING_RATE * (errors[i][j] * trains[i][j]);
		}
		if (++j == 10)
			j = 0;
	}
}

OutputLayer::OutputLayer()
{
	betas = vector<vector<double>>(CLASSES);
	u = vector<vector<double>>(CLASSES);
	y = vector<vector<bool>>(CLASSES);
	gammas = vector<double>(CLASSES);

	srand(time(NULL));

	//generate betas randomly
	for (int i = 0; i < 10; ++i)
	{
		betas[i] = vector<double>(TYO);

		for (int j = 0; j < TYO; ++j)
			betas[i][j] = (rand() % 10 + 1) / 50.0; //0.02 to 0.2
	}
}

void OutputLayer::Reset()
{
	u = vector<vector<double>>(CLASSES);
	y = vector<vector<bool>>(CLASSES);
}

void OutputLayer::ComputeOutput(vector<vector<double>>& synapsesOut)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		u[c] = vector<double>(T);
		y[c] = vector<bool>(T);

		/*
		top N elements in the output of the synapses => input neuron to class 1,2,3,..,N

		therefore the alphas relevant to each output neuron are at rows % 10 where the reminder is the class
		e.g.
		row 0 -> input neuron 0 to class 0
		row 1 -> input neuron 0 to class 1
		...
		*/

		auto alphas = MatrixOps::SumColumnsMod(synapsesOut, c);

		u[c][0] = gammas[c]; //the first potential will always be just the bias

		for (short t = 1; t < T; ++t)
		{
			vector<double> beta = vector<double>(TYO);
			for (short b = t - TYO; b < 0; ++b)
			{
				if (b >= 0)
					beta[t - 1] = (y[c][t] * betas[c][b]);
				else
					beta[t - 1] = 0;
			}

			// sum together alpha, beta, gamma => potential
			u[c][t] = MatrixOps::Sum(beta) + gammas[c] + alphas[t];
			double probability = g(u[c][t]);

			srand(time(NULL));
			probability = probability * 10000;

			if (rand() % 10000 <= probability)
				y[c][t] = 1;
			else
				y[c][t] = 0;
		}
	}
}

vector<vector<double>> OutputLayer::ComputeErrors(unsigned char label) const
{
	vector<vector<double>> diffs = vector<vector<double>>(CLASSES);
	for (short c = 0; c < CLASSES; ++c)
	{
		vector<double> diff = vector<double>(T);
		for (short i = 0; i < T; ++i)
		{
			diff[i] = (label == c ? 1 : 0) - g(u[c][i]);
		}
		diffs[c] = diff;
	}
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

void OutputLayer::UpdateBetas(vector<vector<double>> errors)
{
	short j = 0; //index of 
	for (short i = 0; i < CLASSES; ++i)
	{
		for (short t = 0; t < T; ++t)
		{
			betas[i][t] += LEARNING_RATE * (errors[i][j] * y[i][j]);
		}
		if (++j == 10)
			j = 0;
	}
}

void OutputLayer::UpdateGammas(vector<vector<double>> errors)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		gammas[c] = LEARNING_RATE * MatrixOps::Sum(errors[c]);
	}
}
