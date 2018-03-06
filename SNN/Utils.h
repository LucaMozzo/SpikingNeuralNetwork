#pragma once
#include "stdafx.h"

using std::vector;
using std::string;
using std::pair;

class Utils 
{
public:

	static vector<float> RateEncode(vector<unsigned char>& image);
	static vector<bool> GenerateSpikes(float probability);
	static vector<pair<vector<unsigned char>, unsigned char>> GetTrainingData(int NumberOfImages);
	static vector<pair<vector<unsigned char>, unsigned char>> GetTestData(int NumberOfImages);
};