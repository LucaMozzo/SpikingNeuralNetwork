#include "stdafx.h"
#include "Network.h"
#include "DatabaseOps.h"
#include <algorithm>
#include <vector>
#include <thread>

using std::string;

Network::Network()
{
	inputLayer = InputLayer();
	outputLayer = OutputLayer();
}

char Network::Run(array<unsigned char, NEURONS_IN> image)
{
	// 1. Clear the trains in the output layer
	inputLayer.ResetTrains();
	outputLayer.Reset();

	// 2. Generate the spikes in the input layer
	auto probs = Utils::RateEncode(image);

	for (short i = 0; i < NEURONS_IN; ++i) 
	{
		auto spikes = Utils::GenerateSpikes(probs[i]);
		inputLayer.AddTrain(spikes);
	}

	// 3. Compute Alphas and pass the result to the computation of the output
	auto preProcessedTrains = inputLayer.ApplyAlphas();
	outputLayer.ComputeOutput(preProcessedTrains);

	// 4. Determine the winner based on y
	return outputLayer.ComputeWinner();
}

void Network::Train(short epochs, int trainingImages, vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData)
{
	if(!trainingData)
		for (short epoch = 0; epoch < epochs; ++epoch)
		{
			Utils::PrintLine("Iteration " + std::to_string(epoch));
			auto data = Utils::GetTrainingData(trainingImages);

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

void Network::ImportData(string fileName)
{
	DatabaseOps::ImportData(&inputLayer, &outputLayer, fileName);
}

void Network::ExportData(string fileName)
{
	DatabaseOps::ExportData(&inputLayer, &outputLayer, fileName);
}

int Network::Validate(short testImages)
{
	if (testImages > 10000 || testImages <= 0)
		return 0;

	auto d = Utils::GetTestData(testImages);

	return ValidateDataset(d);
}

int Network::ValidateDataset(vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>& trainingSet)
{
	short correct = 0;
	for (int i = 0; i < trainingSet.size(); ++i)
	{
		const auto res = Run(trainingSet[i].first);

		if (static_cast<int>(res) == static_cast<int>(trainingSet[i].second))
			correct++;
	}

	Utils::PrintLine(std::to_string(correct) + "/" + std::to_string(trainingSet.size()) + " images predicted correctly");

	return correct;
}

int Network::CrossValidate()
{
	auto data = Utils::GetTrainingData(60000);
	int sum = 0;

	vector<vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>> folds{};

	//create folds
	for (short i = 0; i < 10; ++i)
	{
		vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> element(6000);
		std::copy(data.begin()+(i * 6000), data.begin() + ((i + 1) * 6000 - 1), element.begin());
		folds.push_back(element);
	}

	for(short i = 0; i < 10; ++i)
	{
		//prepare the training set
		vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> trainingSet{};
		for (short j = 0; j < 10; ++j)
		{
			if (j == i)
				continue;
			trainingSet.insert(trainingSet.begin(), folds[j].begin(), folds[j].end());
		}

		//train and validate
		ResetNetwork();
		Train(0,0,&trainingSet);
		sum += ValidateDataset(folds[i]);
	}

	std::cout << "\nCross-validation result: " << sum / 10.0 << "\tPercentage: " << sum * 100.0 / 60000.0 << std::endl;
	return sum / 10.0;
}

void Network::ResetNetwork()
{
	inputLayer = InputLayer();
	outputLayer = OutputLayer();
}
