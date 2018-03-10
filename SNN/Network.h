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
	void Train(short epochs, int trainingImages = 60000);
	void Network::ImportData(string fileName = "data.db");
	void Network::ExportData(string fileName = "data.db");
};