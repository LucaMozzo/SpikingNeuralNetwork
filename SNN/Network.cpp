#include "stdafx.h"
#include "Network.h"
#include "DatabaseOps.h"
#include <algorithm>
#include <vector>
#include <thread>
#include <sstream>
#include <fstream>

using std::string;
using std::vector;

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

void Network::ImportData(string fileName)
{
	DatabaseOps::ImportData(&inputLayer, &outputLayer, fileName);
}

void Network::ExportData(string fileName)
{
	DatabaseOps::ExportData(&inputLayer, &outputLayer, fileName);
}

void Network::ImportFile()
{
	std::ifstream wweights("wweights.txt");
	std::ifstream vweights("vweights.txt");
	std::ifstream gweights("gweights.txt");
	std::string line;
	//i is the index of w or v where the value should be put
	for(short i = 0; i < TYI; ++i)
	{
		std::getline(wweights, line);
		std::string buf;
		std::stringstream ss(line);

		int index = 0;
		while (ss >> buf)
			inputLayer.w[index++][i] = atof(buf.c_str());
	}

	wweights.close();
	
	for (short i = 0; i < TYO; ++i)
	{
		std::getline(vweights, line);
		std::string buf;
		std::stringstream ss(line);

		int index = 0;
		while (ss >> buf)
			outputLayer.v[index++][i] = atof(buf.c_str());
	}

	vweights.close();

	//gammas
	for (short i = 0; i < CLASSES; ++i)
	{
		std::getline(gweights, line);
		std::string buf;
		std::stringstream ss(line);

		while (ss >> buf)
			outputLayer.gammas[i] = atof(buf.c_str());
	}

	gweights.close();
}

void Network::ExportFile()
{
	std::ofstream wweights;
	std::ofstream vweights;
	std::ofstream gweights;
	wweights.open("wweights_out.txt");
	for (short i = 0; i < TYI; ++i)
	{
		for(int index = 0; index < NEURONS_IN*CLASSES; ++index)
			wweights << inputLayer.w[index][i] << " ";
		wweights << "\n";
	}
	wweights.flush();
	wweights.close();

	vweights.open("vweights_out.txt");
	for (short i = 0; i < TYO; ++i)
	{
		for (int index = 0; index < CLASSES; ++index)
			vweights << outputLayer.v[index][i] << " ";
		vweights << "\n";
	}
	vweights.flush();
	vweights.close();

	gweights.open("gweights_out.txt");
	for (int index = 0; index < CLASSES; ++index)
		gweights << outputLayer.gammas[index] << "\n";
	gweights.flush();
	gweights.close();
}

int Network::ValidateDataset(vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>& trainingSet)
{
	int correct = 0;
	for (int i = 0; i < trainingSet.size(); ++i)
	{
		const auto res = Run(trainingSet[i].first);

		if (static_cast<int>(res) == static_cast<int>(trainingSet[i].second))
			correct++;
		/*else
			Utils::PrintLine("Expected " + std::to_string(trainingSet[i].second) + " but got " + std::to_string(res));*/
	}

	Utils::PrintLine(std::to_string(correct) + "/" + std::to_string(trainingSet.size()) + " images predicted correctly (" + std::to_string(correct/(float)trainingSet.size()*100) + "%)");

	return correct;
}

int Network::CrossValidate()
{
	auto data = Utils::GetTrainingData<0>(60000);
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
		Train<0>(0,0,&trainingSet);
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
