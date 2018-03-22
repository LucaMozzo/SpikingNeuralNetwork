#pragma once
#include "stdafx.h"
#include <mutex>

using std::array;
using std::string;
using std::pair;

class Utils 
{
private:

	static std::mutex lock;
	static float RaisedCosine(int time, int mean, float stddev);

public:

	static array<float, NEURONS_IN> RateEncode(array<unsigned char, NEURONS_IN>& image);
	static array<bool, T> GenerateSpikes(float probability);
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTrainingData(int NumberOfImages);
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTestData(int NumberOfImages);
	static array<float, T> GenerateBasisMatrix(short meanOffset);
	static array<array<double, Ka>, TYI> GenerateAlphaBasis();
	static array<array<double, Kb>, TYO> GenerateBetaBasis();
	static void PrintLine(string&& str);
};