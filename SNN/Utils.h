#pragma once
#include "stdafx.h"

using std::vector;
using std::string;
using std::pair;

class Utils 
{
public:

	static float* RateEncode(unsigned char* image);
	static bool* GenerateSpikes(float probability);
	static pair<unsigned char*, unsigned char>* GetTrainingData(int NumberOfImages);
	static pair<unsigned char*, unsigned char>* GetTestData(int NumberOfImages);
};