#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"

class Network {
protected:

	InputLayer inputLayer;
	OutputLayer outputLayer;

public:

	Network();
	char Run(vector<unsigned char> image);
	void Train();
};