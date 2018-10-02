#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "Utils.h"

/**
The Network class contains the methods to perform all the possible operations on the network
*/
class Network {
protected:

	InputLayer inputLayer; /**< The input layer */
	OutputLayer outputLayer; /**< The output layer */

	/**
	To reset both layers after processing every image
	*/
	void ResetNetwork();

	int epoch = 0;

public:

	/**
	The constructor instantiates the layers
	*/
	Network();
	/**
	Run the network on an image and return the predicted class
	@param image The input image to be classified
	@return The predicted class of the image
	*/
	char Run(array<unsigned char, NEURONS_IN> image);
	/**
	Trains the network
	@param FILTER_SIZE The number of elements in the filter
	@param epochs The number of iterations to perform
	@param trainingImages The number of images to train the network on
	@param trainingData The data to train the network on - if it's not nullptr, it overrides the trainingImages parameter
	@param filter The filter to limit the labels in the training - if nullptr, all 10 digits are considered
	@param maxImagesPerLabel The maximum of images for each digit
	*/
	template <std::size_t FILTER_SIZE>
	void Train(short epochs, int trainingImages = 60000, vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData = nullptr, array<unsigned char, FILTER_SIZE>* filter = nullptr, int maxImagesPerLabel = 0);
	/**
	Import network weights from a database file
	@param fileName The file name of the database from which import the weights
	*/
	void ImportData(string fileName = "data.db");
	/**
	Export network weights to a database file
	@param fileName The file name of the database to which export the weights
	*/
	void ExportData(string fileName = "data.db");
	/**
	Import network weights from a txt file
	*/
	void ImportFile();
	/**
	Export network weights to a txt file
	*/
	void ExportFile();
	/**
	Validates the network on any data
	@param FILTER_SIZE The number of elements in the filter
	@param testImages The number of images used for validation. The range should be [1 10000] for the test set and [1 60000] for the training one
	@param testSet If true the test set is used, otherwise the training set
	@param filter The filter to limit the labels in the training - if nullptr, all 10 digits are considered
	@param maxImagesPerLabel The maximum of images for each digit
	@return The number of correct classifications
	*/
	template <std::size_t FILTER_SIZE>
	int Validate(int testImages = 10000, bool testSet = true, array<unsigned char, FILTER_SIZE>* filter = NULL, int maxImagesPerLabel = 0);
	/**
	Validates the network on the given images
	@param trainingSet The images that should be used to validate the network on
	@return The number of correct classifications
	*/
	int ValidateDataset(vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>& trainingSet);
	/**
	Performs 10-folds cross-validation on the entire training set - single epoch
	@returns The average of correct classification - Max 6000
	*/
	int CrossValidate();

	int TrainVal(int epochs, int imagesPerLabel, int validationImages, bool collectData = false);
};

template <std::size_t FILTER_SIZE>
void Network::Train(short epochs, int trainingImages,
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>* trainingData, array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel)
{
	if (!trainingData)
		for (short e = 0; e < epochs; ++epoch, ++e)
		{
			Utils::PrintLine("Epoch " + std::to_string(epoch));
			auto data = Utils::GetTrainingData(trainingImages, filter, maxImagesPerLabel);

			for (int i = 0; i < data.size(); ++i)
			{
				auto result = Run(data[i].first);
				if (result == data[i].second)
				{
					auto errors = outputLayer.ComputeErrors(data[i].second);
					auto max_z_j = outputLayer.ComputeMaxZj();

					inputLayer.UpdateAlphas(errors, outputLayer.z[data[i].second], max_z_j);
					outputLayer.UpdateBetas(errors, max_z_j);
					outputLayer.UpdateGammas(errors, max_z_j);
				}
			}
		}
	else
	{
		for (short e = 0; e < epochs; ++epoch, ++e)
		{
			Utils::PrintLine("Epoch " + std::to_string(epoch) + " (" + std::to_string(trainingData->size()) + " images)");
			for (auto& img : *trainingData)
			{
				auto result = Run(img.first);
				if (result == img.second)
				{
					auto errors = outputLayer.ComputeErrors(img.second);
					const auto max_z_y = outputLayer.ComputeMaxZj();

					inputLayer.UpdateAlphas(errors, outputLayer.z[img.second], max_z_y);
					outputLayer.UpdateBetas(errors, max_z_y);
					outputLayer.UpdateGammas(errors, max_z_y);
				}
			}
		}
	}
}

template <std::size_t FILTER_SIZE>
int Network::Validate(int testImages, bool testSet, array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel)
{
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> d;
	if (testSet)
		d = Utils::GetTestData(testImages, filter, maxImagesPerLabel);
	else
	{
		d = Utils::GetTrainingData(testImages, filter, maxImagesPerLabel);
	}

	return ValidateDataset(d);
}
