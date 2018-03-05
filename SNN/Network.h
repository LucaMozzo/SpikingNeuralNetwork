#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"

class Network {
protected:

	InputLayer* inputLayer;
	OutputLayer* outputLayer;

public:

	Network();
	~Network();
	char Run();
	void Train();
};