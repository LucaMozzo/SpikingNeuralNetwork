#include "stdafx.h"
#include "Layer.h"

InputLayer::InputLayer()
{
	trains = array<array<bool, T>, NEURONS_IN>();
	alphas = array<array<double, TYI>, CLASSES*NEURONS_IN>();

	//generate alphas randomly
	for (int i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		alphas[i] = array<double, TYI>();

		for (int j = 0; j < TYI; ++j)
			alphas[i][j] = (rand() % 10 + 1) / 10.0;
	}
}

void InputLayer::AddTrain(array<bool, T>& train)
{
	trains[index++] = train;
}

void InputLayer::ResetTrains()
{
	index = 0;
	trains = array<array<bool, T>, NEURONS_IN>();
}

array<array<double, T-1>, CLASSES*NEURONS_IN> InputLayer::ApplyAlphas() const
{
	short j = 0; //index for the train
	array<array<double,T-1>, CLASSES*NEURONS_IN> result = array<array<double, T-1>, CLASSES*NEURONS_IN>();
	for (short c = 0; c < NEURONS_IN*CLASSES; ++c)
	{
		if (c % CLASSES == 0 && c > 0)
			++j;
		result[c] = MatrixOps::Conv(trains[j], alphas[c]);
	}
	return result;
}

void InputLayer::UpdateAlphas(array<array<double, T>, CLASSES>& errors)
{
	short j = 0; //index for the train
	for (short c = 0; c < CLASSES*NEURONS_IN; ++c)
	{
		if (c % CLASSES == 0 && c > 0)
			++j;

		//compute error
		double tot = 0;
		for (short t = 0; t < T; ++t)
			tot += errors[c%10][t] * trains[j][t];
		tot *= LEARNING_RATE;

		//apply it to every member of alpha
			for (short t = 0; t < TYI; ++t)
			{
				alphas[c][t] += tot;
			}
	}
}

OutputLayer::OutputLayer()
{
	betas = array<array<double, TYO>, CLASSES>();
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
	gammas = array<double, CLASSES>();

	//srand(time(NULL));

	//generate betas and gammas randomly
	for (int i = 0; i < 10; ++i)
	{
		betas[i] = array<double, TYO>();

		for (int j = 0; j < TYO; ++j)
			betas[i][j] = (rand() % 10 + 1) / 10.0; //0.02 to 0.2
		gammas[i] = (rand() % 10 + 1) / 10.0;;
	}
}

void OutputLayer::Reset()
{
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
}

void OutputLayer::ComputeOutput(array<array<double, T-1>, CLASSES*NEURONS_IN>& synapsesOut)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		//srand(time(NULL));
		u[c] = array<double, T>();
		y[c] = array<bool, T>();

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
		//compute spiking
		double probability = g(u[c][0]);
		probability = probability * 10000;

		if (rand() % 10000 <= probability)
			y[c][0] = 1;
		else
			y[c][0] = 0;

		for (short t = 1; t < T; ++t)
		{
			array<double, TYO> beta = array<double, TYO>();
			short yIndex = t - TYO;
			for (short b = 0; b < TYO; ++b)
			{
				if (yIndex++ >= 0)
					beta[b] = y[c][yIndex-1] * betas[c][b];
				else
					beta[b] = 0;
			}

			// sum together alpha, beta, gamma => potential
			u[c][t] = MatrixOps::template Sum<TYO>(beta) + gammas[c] + alphas[t-1];

			//compute spiking
			double probability = g(u[c][t]);
			probability = probability * 10000;

			if (rand() % 10000 <= probability)
				y[c][t] = 1;
			else
				y[c][t] = 0;
		}
	}
}

array<array<double, T>, CLASSES> OutputLayer::ComputeErrors(unsigned char label) const
{
	array<array<double, T>, CLASSES> diffs = array<array<double, T>, CLASSES>();
	for (short c = 0; c < CLASSES; ++c)
	{
		array<double, T> diff = array<double, T>();
		for (short t = 0; t < T; ++t)
		{
			diff[t] = (label == c ? 1 : 0) - g(u[c][t]);
		}
		diffs[c] = diff;
	}
	return diffs;
}

char OutputLayer::ComputeWinner() const
{
	char bestIndex = 0;
	char bestSpikes = 0;

	for (short i = 0; i < CLASSES; ++i) 
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

void OutputLayer::UpdateBetas(array<array<double, T>, CLASSES>& errors)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		//compute error
		double tot = 0;
		for (short t = 0; t < T; ++t)
			tot += errors[c][t] * y[c][t];

		tot *= LEARNING_RATE;

		//apply it
		for (short t = 0; t < TYO; ++t)
		{
			betas[c][t] += tot;
		}
	}
}

void OutputLayer::UpdateGammas(array<array<double, T>, CLASSES>& errors)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		auto sum = MatrixOps::Sum(errors[c]);
		gammas[c] += LEARNING_RATE * sum;
	}
}
