#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"

class Network {
protected:

	InputLayer inputLayer;
	OutputLayer outputLayer;
	HiddenLayer middleLayer;

	void ResetNetwork();

public:

	Network();
	char Run(array<unsigned char, NEURONS_IN> image);
	void Train(short epochs, int trainingImages = 60000, vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData = nullptr, bool validateAfterEpoch = false);
	void ImportData(string fileName = "data.db");
	void ExportData(string fileName = "data.db");
	int Validate(short testImages = 10000);
	int ValidateDataset(vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>& trainingSet);
	int CrossValidate();
};