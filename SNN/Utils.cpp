#include "stdafx.h"
#include "Utils.h"
#include "Constants.h"
#include <opencv2\opencv.hpp>

# define M_PI 3.14159265358979323846L

using namespace cv;

std::mutex Utils::lock;


float Utils::RaisedCosine(int time, int mean, float stddev)
{
	return 0.45 * (1 + cos((time - mean) / stddev * M_PI)) + 0.1;
}

array<array<double, Ka>, TYI> Utils::GenerateAlphaBasis()
{
	array<array<double, Ka>, TYI> result = array<array<double, Ka>, TYI>();

	if(BASIS_FUNCTION == RAISED_COSINE)
	{
		for(short i = 0; i < 2*Ka; i+=2)
		{
			result[i / 2] = array<double, Ka>();
			for (short j = 0; j < TYI; ++j)
				result[i / 2][j] = RaisedCosine(j, i, Ka);
		}
	}
	else
	{
		//binary
		for(short i = 0; i < Ka; ++i)
		{
			for(short j = 0; j < Ka; ++j)
			{
				result[i][j] = i == j ? 1 : 0;
			}
		}
	}
	return result;
}

array<array<double, Kb>, TYO> Utils::GenerateBetaBasis()
{
	array<array<double, Kb>, TYO> result = array<array<double, Kb>, TYO>();

	if(BASIS_FUNCTION == RAISED_COSINE)
	{
		for (short i = 0; i < 2 * Kb; i += 2)
		{
			result[i / 2] = array<double, Kb>();
			for (short j = 0; j < TYO; ++j)
				result[i / 2][j] = RaisedCosine(j, i, Kb);
		}
	}
	else
	{
		//binary
		for(short i = 0; i < Ka; ++i)
		{
			for(short j = 0; j < Ka; ++j)
			{
				result[i][j] = i == j ? 1 : 0;
			}
		}
	}
	return result;
}

/*
 Return a probability of spiking for every pixel
*/
array<float, NEURONS_IN> Utils::RateEncode(array<unsigned char, NEURONS_IN>& image)
{
	//auto imagePixels = GetPixelIntensities(imagePath);
	array<float, NEURONS_IN> probabilities = array<float, NEURONS_IN>();
	
	for (int i = 0; i < 784; ++i)
	{
		probabilities[i] = image[i] * (P_RANGE.second - P_RANGE.first) / 255 + P_RANGE.first;
	}
	return probabilities;
}

/*
 Return a array of size T that contains the train of spikes based on probability
*/
array<bool,T> Utils::GenerateSpikes(float probability)
{
	array<bool,T> train = array<bool,T>();

	//random seed

	probability = probability * 10000;

	for (int i = 0; i < T; ++i)
		if (rand() % 10000 <= probability)
			train[i] = 1;
		else
			train[i] = 0;

	return train;
}

int Utils::ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void Utils::PrintLine(string&& str)
{
	auto now = std::chrono::system_clock::now();
	auto now_c = std::chrono::system_clock::to_time_t(now);

#pragma warning(suppress : 4996)
	std::cout << std::put_time(std::localtime(&now_c), "%c") << " - " << str << std::endl;
}

double Utils::GetStepSize(pair<double, double>& range)
{
	return (range.second - range.first) / (2 ^ (PRECISION - 1));
}