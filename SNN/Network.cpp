#include "stdafx.h"
#include "Network.h"
#include "Utils.h"

Network::Network()
{
	inputLayer = new InputLayer();
	outputLayer = new OutputLayer();
}

Network::~Network()
{
	delete inputLayer;
	delete outputLayer;
}

char Network::Run(unsigned char* image)
{
	// 1. Clear the trains in the output layer
	inputLayer->ResetTrains();
	outputLayer->Reset();

	// 2. Generate the spikes in the input layer
	auto probs = Utils::RateEncode(image);

	for (short i = 0; i < NEURONS_IN; ++i) 
	{
		auto spikes = Utils::GenerateSpikes(probs[i]);
		inputLayer->AddTrain(spikes);
	}

	// 3. Compute Alphas and pass the result to the computation of the output
	inputLayer->ApplyAlphas();

	// 4. Determine the winner based on y
	return 0;
}
