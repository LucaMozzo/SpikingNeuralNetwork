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
	char Run(array<unsigned char, NEURONS_IN> image);
	void Train(short epochs, int trainingImages = 60000);
	void ImportData(string fileName = "data.db");
	void ExportData(string fileName = "data.db");
	int Validate(bool verbose, short testImages = 10000);
};