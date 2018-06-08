#include "Layer.h"
#include "Utils.h"

InputLayer::InputLayer()
{
	trains = array<array<bool, T>, NEURONS_IN>();
	w = array<array<double, TYI>, CLASSES*NEURONS_IN>();

	//generate w randomly
	for (int i = 0; i < CLASSES*NEURONS_IN; ++i)
	{
		w[i] = array<double, TYI>();

		for (int j = 0; j < TYI; ++j)
			w[i][j] = (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 2))) - 1;
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

void InputLayer::UpdateAlphas(array<array<double, TYI>, CLASSES*NEURONS_IN>& gradients)
{
	for (short c = 0; c < CLASSES*NEURONS_IN; ++c)
	{
		//apply it to every member of w
		for (short t = 0; t < TYI; ++t)
		{
			w[c][t] += LEARNING_RATE * gradients[c][t];
		}
	}
}

OutputLayer::OutputLayer()
{
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
	q = array<double, T>();
	h = array<double, T>();
	gammas = array<double, CLASSES>();

	//generate v and gammas randomly
	for (int i = 0; i < 10; ++i)
	{
		gammas[i] = (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / 2))) - 1;
	}
}

void OutputLayer::Reset()
{
	u = array<array<double, T>, CLASSES>();
	y = array<array<bool, T>, CLASSES>();
	q = array<double, T>();
	h = array<double, T>();
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

		therefore the w relevant to each output neuron are at rows % 10 where the reminder is the class
		e.g.
		row 0 -> input neuron 0 to class 0
		row 1 -> input neuron 0 to class 1
		...
		*/

		auto alphas = MatrixOps::SumColumns(synapsesOut, c);

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
			// sum together alpha, beta, gamma => potential
			u[c][t] = gammas[c] + alphas[t - 1];

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

char OutputLayer::ComputeWinner() const
{
	for (short t = 0; t < T; ++t)
	{
		for (short i = 0; i < CLASSES; ++i)
			if (y[i][t] == 1)
				return i;
	}
	return 0; //no spike
}


array<double, T> OutputLayer::FTSProbability(char label)
{
	auto probabilities = array<double, T>();

	for (short t = 0; t < T; ++t)
	{
		double prob = 0;
		bool flag = true;
		for (int i = 0; i < CLASSES; ++i)
		{
			//skip the correct class
			if (i == label)
				continue;

			for (short t1 = 0; t1 <= t; ++t1)
			{
				if (flag)
				{
					prob = (1 - g(u[i][t1])) * g(u[label][t]);
					flag = !flag;
				}
				else
					prob *= (1 - g(u[i][t1])) * g(u[label][t]);

				for (short t2 = 0; t2 <= t - 1; ++t2)
				{
					prob *= (1 - g(u[label][t2]));
				}
			}
		}

		probabilities[t] = prob;
	}

	return probabilities;
}

void OutputLayer::ComputeQ(char label)
{
	auto p = FTSProbability(label);

	auto psum = MatrixOps::Sum<T>(p);


	for (short t = 0; t < T; ++t)
	{
		q[t] = p[t] / psum;
	}
}

void OutputLayer::ComputeH(char label)
{

	for (short t = 0; t < T; ++t)
	{
		double sum = 0;
		for (short t1 = t; t1 < T; ++t1)
			sum += q[t1];

		h[t] = sum;
	}
}

array<array<double, TYI>, CLASSES*NEURONS_IN> OutputLayer::GradientW(char label, array<array<bool, T>, NEURONS_IN>& x, array<array<double, Ka>, TYI>& basis)
{
	auto gradients = array<array<double, TYI>, CLASSES*NEURONS_IN>();
	auto basisTranspose = MatrixOps::Transpose(basis);

	int trainIndex = 0;
	for (short j = 0; j < CLASSES*NEURONS_IN; ++j)
	{
		if (j % 10 == 0 && j > 0)
			++trainIndex;

		array<double, TYI> sum;

		//initalize with 0s
		for (short t = 0; t < TYI; ++t)
			sum[t] = 0;

		for (short t = 0; t < T; ++t)
		{
			//compute the window to be considered
			array<double, TYI> trainWindow{};
			short index = 0;
			for (short i = t - 1; i > t - TYI; --i)
			{
				if (i < 0)
				{
					trainWindow[index++] = 0;
				}
				else
				{
					trainWindow[index++] = x[trainIndex][i];
				}
			}

			if (MatrixOps::Sum(trainWindow) != 0) {

				auto dotProd = MatrixOps::Dot(basisTranspose, trainWindow);

				//check the two cases
				if (j % 10 == label)
					//if the current output neuron = the label
					MatrixOps::Multiply(h[t] * g(u[j % 10][t]) - q[t], dotProd);
				else
					MatrixOps::Multiply(h[t] * g(u[j % 10][t]), dotProd);
				sum = MatrixOps::SumArrays(sum, dotProd);
			}

		}
		MatrixOps::Multiply(-1, sum);
		gradients[j] = sum;
	}

	return gradients;
}

array<double, CLASSES> OutputLayer::GradientGamma(char label)
{
	auto gradients = array<double, CLASSES>();

	for (short i = 0; i < CLASSES; ++i)
	{
		double sum = 0;
		for (short t = 0; t < T; ++t)
		{
			if (i == label)
				//if the current output neuron = the label
				sum += h[t] * g(u[i][t]) - q[t];
			else
				sum += h[t] * g(u[i][t]);
		}
		gradients[i] = -sum;
	}

	return gradients;
}

void OutputLayer::UpdateGammas(array<double, CLASSES>& gradients)
{
	for (short c = 0; c < CLASSES; ++c)
	{
		gammas[c] += LEARNING_RATE * gradients[c];
	}
}
