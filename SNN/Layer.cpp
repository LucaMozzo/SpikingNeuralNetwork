#include "stdafx.h"
#include "Layer.h"

InputLayer::InputLayer()
{
	trains = vector<vector<bool>>(NEURONS_IN);
	alphas = vector<vector<double>>(NEURONS_IN*CLASSES);

	//srand(time(NULL));

	//generate alphas randomly
	for (int i = 0; i < 7840; ++i)
	{
		alphas[i] = vector<double>(TYI);

		for (int j = 0; j < TYI; ++j)
			alphas[i][j] = (rand() % 10 + 1) / 10.0;
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
	short j = 0; //index for the train
	vector<vector<double>> result = vector<vector<double>>(CLASSES*NEURONS_IN);
	for (short c = 0; c < NEURONS_IN*CLASSES; ++c)
	{
		if (c % CLASSES == 0 && c > 0)
			++j;
		result[c] = MatrixOps::Conv(trains[j], alphas[c]);
	}
	/*for (short c = 0; c < CLASSES; ++c)
	{
		for (short i = 0; i < NEURONS_IN; ++i)
		{
			result[c*NEURONS_IN + i] = MatrixOps::Conv(trains[i], alphas[c*NEURONS_IN + i]);
		}
	}*/
	return result;
}

void InputLayer::UpdateAlphas(vector<vector<double>>& errors)
{
	/*for (short c = 0; c < CLASSES; ++c)
	{
		for (short i = 0; i < NEURONS_IN; ++i)
		{
			//compute error
			double tot = 0;
			for (short t = 0; t < T; ++t)
				tot += errors[c][t] * trains[i][t];
			tot *= LEARNING_RATE;

			//apply it to every member of alpha
			//if (tot > 0)
				for (short t = 0; t < TYI; ++t)
				{
					alphas[c*NEURONS_IN+i][t] += tot;
				}
		}
	}*/
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
	betas = vector<vector<double>>(CLASSES);
	u = vector<vector<double>>(CLASSES);
	y = vector<vector<bool>>(CLASSES);
	gammas = vector<double>(CLASSES);

	//srand(time(NULL));

	//generate betas and gammas randomly
	for (int i = 0; i < 10; ++i)
	{
		betas[i] = vector<double>(TYO);

		for (int j = 0; j < TYO; ++j)
			betas[i][j] = (rand() % 10 + 1) / 10.0; //0.02 to 0.2
		gammas[i] = (rand() % 10 + 1) / 10.0;;
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
		//srand(time(NULL));
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
		//compute spiking
		double probability = g(u[c][0]);
		probability = probability * 10000;

		int rnd = Random::random();
		if (/*rand() % 10000*/rnd*1000 <= probability)
			y[c][0] = 1;
		else
			y[c][0] = 0;

		for (short t = 1; t < T; ++t)
		{
			vector<double> beta = vector<double>(TYO);
			short yIndex = t - TYO;
			for (short b = 0; b < 2; ++b)
			{
				if (yIndex++ >= 0)
					beta[b] = y[c][yIndex-1] * betas[c][b];
				else
					beta[b] = 0;
			}

			// sum together alpha, beta, gamma => potential
			u[c][t] = MatrixOps::Sum(beta) + gammas[c] + alphas[t-1];

			//compute spiking
			double probability = g(u[c][t]);
			probability = probability * 10000;

			if (/*rand() % 10000*/Random::random() * 1000 <= probability)
				y[c][t] = 1;
			else
				y[c][t] = 0;
		}
	}
}

vector<vector<double>> OutputLayer::ComputeErrors(unsigned char label) const
{
	double tot = 0;
	vector<vector<double>> diffs = vector<vector<double>>(CLASSES);
	for (short c = 0; c < CLASSES; ++c)
	{
		vector<double> diff = vector<double>(T);
		for (short t = 0; t < T; ++t)
		{
			diff[t] = (label == c ? 1 : 0) - g(u[c][t]);
			//tot += diff[t];
		}
		diffs[c] = diff;
	}
	//std::cout << "Err " << tot << std::endl;
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

void OutputLayer::UpdateBetas(vector<vector<double>> errors)
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

void OutputLayer::UpdateGammas(vector<vector<double>> errors)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		auto sum = MatrixOps::Sum(errors[c]);
		gammas[c] += LEARNING_RATE * sum;
	}
}
