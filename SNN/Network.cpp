#include "stdafx.h"
#include "Network.h"
#include "DatabaseOps.h"

using std::string;

Network::Network()
{
	srand(time(NULL));
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

void Network::Train(short epochs, int trainingImages)
{
	for (short epoch = 0; epoch < epochs; ++epoch)
	{
		std::cout << "Iteration" << std::endl;
		auto data = Utils::GetTrainingData(trainingImages);

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

void Network::ImportData(string fileName)
{
	DatabaseOps::ImportData(&inputLayer, &outputLayer, fileName);
}

void Network::ExportData(string fileName)
{
	DatabaseOps::ExportData(&inputLayer, &outputLayer, fileName);
}

int Network::Validate(bool verbose, short testImages)
{
	if (testImages > 10000 || testImages <= 0)
		return 0;

	auto d = Utils::GetTestData(testImages);

	short correct = 0;
	for (int i = 0; i < d.size(); ++i)
	{
		const auto res = Run(d[i].first);
		if(verbose)
			std::cout << "Predicted: " << static_cast<int>(res) << "\tWas: " << static_cast<int>(d[i].second) << std::endl;

		if (static_cast<int>(res) == static_cast<int>(d[i].second))
			correct++;
	}

	std::cout << "\n" << correct << "/" << d.size() << " images predicted correctly" << std::endl;
	
	return correct;
}
