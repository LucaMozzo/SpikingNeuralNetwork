#include "stdafx.h"
#include "Network.h"
#include "DatabaseOps.h"

Network::Network()
{
	inputLayer = InputLayer();
	outputLayer = OutputLayer();
}

char Network::Run(vector<unsigned char> image)
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

void Network::Train(short epochs, int trainingImages)
{
	for (short epoch = 0; epoch < epochs; ++epoch)
	{
		auto data = Utils::GetTestData(trainingImages);

		for (int i = 0; i < trainingImages; ++i)
		{
			char result = Run(data[i].first);
			auto errors = outputLayer.ComputeErrors(data[i].second);
			inputLayer.UpdateAlphas(errors);

			outputLayer.UpdateBetas(errors);
			outputLayer.UpdateGammas(errors);
		}
	}
}

void Network::ImportData()
{
	DatabaseOps::ImportData(&inputLayer, &outputLayer);
}
