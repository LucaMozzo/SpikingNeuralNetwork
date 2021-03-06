#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"
#include "Constants.h"

InputLayer::InputLayer()
{
	trains = array<array<bool, T>, NEURONS_IN>();
	w = array<array<double, TYI>, CLASSES*NEURONS_IN>();

	//generate w randomly
	for (int i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		w[i] = array<double, TYI>();

		for (int j = 0; j < TYI; ++j)
			w[i][j] = (rand() % 10 + 1) / 10.0;
	}

	basis = Utils::GenerateAlphaBasis();
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

array<array<double, T-1>, CLASSES*NEURONS_IN> InputLayer::ApplyAlphas()
{
	short j = 0; //index for the train
	array<array<double,T-1>, CLASSES*NEURONS_IN> result = array<array<double, T-1>, CLASSES*NEURONS_IN>();
	for (short c = 0; c < NEURONS_IN*CLASSES; ++c)
	{
		if (c % CLASSES == 0 && c > 0)
			++j;
		result[c] = MatrixOps::Conv(trains[j], MatrixOps::Dot(basis, w[c]));
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
		//double tot = 0;
		array<double, TYI> err{};
		for (short t = 0; t < T; ++t)
		{
			array<double, TYI> trainWindow{ };
			short index = 0;
			for(short i = t-1; i > t-TYI; --i)
			{
				if (i < 0)
					trainWindow[index++] = 0;
				else 
				{
					trainWindow[index++] = trains[j][i];
				}
			}

			auto basisT = MatrixOps::Transpose(basis);
			auto tmp = MatrixOps::Dot(basisT, trainWindow);
			MatrixOps::Multiply(errors[c%10][t], tmp);
			err = MatrixOps::SumArrays(err, tmp);
		}

		MatrixOps::Multiply(LEARNING_RATE, err);

		
		//apply it to every member of alpha
		for (short t = 0; t < TYI; ++t)
		{
			w[c][t] += err[t];
		}
	}
}

OutputLayer::OutputLayer()
{
	v = array<array<double, TYO>, CLASSES>();
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
	gammas = array<double, CLASSES>();

	//srand(time(NULL));

	//generate v and gammas randomly
	for (int i = 0; i < 10; ++i)
	{
		v[i] = array<double, TYO>();

		for (int j = 0; j < TYO; ++j)
			v[i][j] = (rand() % 10 + 1) / 10.0; //0.02 to 0.2
		gammas[i] = (rand() % 10 + 1) / 10.0;;
	}

	basis = Utils::GenerateBetaBasis();
}

void OutputLayer::Reset()
{
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
}

void OutputLayer::ComputeOutput(array<array<double, T-1>, CLASSES*NEURONS_IN>& synapsesOut, signed char label)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		u[c] = array<double, T>();
		y[c] = array<bool, T>();

		/*
		top N elements in the output of the synapses => input neuron to class 1,2,3,..,N

		therefore the w relevant to each output neuron are at rows % 10 where the reminder is the class
		e.g.
		row 0 -> input neuron 0 to class 0
		row 1 -> input neuron 0 to class 1
		...
		*/

		auto alphas = MatrixOps::SumColumns(synapsesOut, c);

		u[c][0] = gammas[c]; //the first potential will always be just the bias
		double probability;

		if(label == -1)
		{
			//prediction, compute spiking
			probability = g(u[c][0]);
			probability = probability * 10000;

			if (rand() % 10000 <= probability)
				y[c][0] = 1;
			else
				y[c][0] = 0;
		}
		else
		{
			//training, preset y
			for(char i = 0; i < CLASSES; ++i)
				for(int j = 0; j < T; ++j)
					if(i == label)
						y[i][j] = 1;
					else
						y[i][j] = 0;
		}

		for (short t = 1; t < T; ++t)
		{
			array<double, TYO> beta = array<double, TYO>();
			short yIndex = t - 1; // I start considering the output at time t-tauy'
			for (short b = 0; b < TYO; ++b)
			{
				if (yIndex-- >= 0) //if the window is within the boundaries
					beta[b] = y[c][yIndex+1] * MatrixOps::Dot(basis, v[c])[b]; //use the formula to multiply beta * y
				else
					beta[b] = 0; //if the window is out of the boundaries, y is assumed to be 0
			}

			// sum together alpha, beta, gamma => potential
			u[c][t] = MatrixOps::template Sum<TYO>(beta) + gammas[c] + alphas[t-1];

			if(label == -1)
			{
				//compute spiking
				probability = g(u[c][t]);
				probability = probability * 10000;

				if (rand() % 10000 <= probability)
					y[c][t] = 1;
				else
					y[c][t] = 0;
			}
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
			diff[t] = (label == c ? CORRECT_PATTERN[t] : 0) - g(u[c][t]);
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
		array<double, TYO> err{};
		for (short t = 0; t < T; ++t)
		{
			array<double, TYO> trainWindow{};
			short index = 0;
			for (short i = t - TYI; i < t - 1 ; ++i)
			{
				if (i < 0)
					trainWindow[index++] = 0;
				else 
				{
					trainWindow[index] = y[c][i];
					++index;
				}
			}

			auto basisT = MatrixOps::Transpose(basis);
			auto tmp = MatrixOps::Dot(basisT, trainWindow);
			MatrixOps::Multiply(errors[c][t], tmp);
			err = MatrixOps::SumArrays(err, tmp);
		}

		MatrixOps::Multiply(LEARNING_RATE, err);


		//apply it
		for (short t = 0; t < TYO; ++t)
		{
			v[c][t] += err[t];
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
