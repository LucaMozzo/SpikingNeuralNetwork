#pragma once
#include "stdafx.h"

using std::array;
using std::string;
using std::pair;

class Utils 
{
public:

	static array<float, NEURONS_IN> RateEncode(array<unsigned char, NEURONS_IN>& image);
	static array<bool, T> GenerateSpikes(float probability);
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTrainingData(int NumberOfImages);
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTestData(int NumberOfImages);
};