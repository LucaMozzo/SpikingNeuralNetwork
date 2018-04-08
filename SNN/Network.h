#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"

class Network {
protected:

	InputLayer inputLayer;
	OutputLayer outputLayer;

	void ResetNetwork();

public:

	Network();
	char Run(array<unsigned char, NEURONS_IN> image);
	template <std::size_t FILTER_SIZE>
	void Train(short epochs, int trainingImages = 60000, vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData = nullptr, array<unsigned char, FILTER_SIZE>* filter = NULL);
	void ImportData(string fileName = "data.db");
	void ExportData(string fileName = "data.db");
	template <std::size_t FILTER_SIZE>
	int Validate(int testImages = 10000, bool testSet = true, array<unsigned char, FILTER_SIZE>* filter = NULL);
	int ValidateDataset(vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>& trainingSet);
	int CrossValidate();
};

template <std::size_t FILTER_SIZE>
void Network::Train(short epochs, int trainingImages,
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData, array<unsigned char, FILTER_SIZE>* filter)
{
	if (!trainingData)
		for (short epoch = 0; epoch < epochs; ++epoch)
		{
			Utils::PrintLine("Iteration " + std::to_string(epoch));
			auto data = Utils::GetTrainingData(trainingImages, filter);

			for (int i = 0; i < trainingImages; ++i)
			{
				Run(data[i].first);
				auto errors = outputLayer.ComputeErrors(data[i].second);
				inputLayer.UpdateAlphas(errors);

				outputLayer.UpdateBetas(errors);
				outputLayer.UpdateGammas(errors);
			}
		}
	else
	{
		for (auto& img : *trainingData)
		{
			Run(img.first);
			auto errors = outputLayer.ComputeErrors(img.second);
			inputLayer.UpdateAlphas(errors);

			outputLayer.UpdateBetas(errors);
			outputLayer.UpdateGammas(errors);
		}
	}
}

template <std::size_t FILTER_SIZE>
int Network::Validate(int testImages, bool testSet, array<unsigned char, FILTER_SIZE>* filter)
{
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> d;
	if (testSet)
		d = Utils::GetTestData(testImages, filter);
	else
	{
		d = Utils::GetTrainingData(testImages, filter);
	}

	return ValidateDataset(d);
}
