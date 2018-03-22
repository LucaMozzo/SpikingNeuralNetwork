#include "stdafx.h"
#include "Utils.h"
#include "Constants.h"
#include <opencv2\opencv.hpp>
#include <thread>
#include <algorithm>
#include <algorithm>

# define M_PI 3.14159265358979323846L

using namespace cv;

std::mutex Utils::lock;


float Utils::RaisedCosine(int time, int mean, float stddev)
{
	return 0.5 * (1 + cos((time - mean) / stddev * M_PI)) + 0.1;
}

array<float, T> Utils::GenerateBasisMatrix(short meanOffset)
{
	array<float, T> result{};

	for (int t = 0; t < T; ++t)
		result[t] = RaisedCosine(t, T + meanOffset, 10);

	return result;
}

array<array<double, Ka>, TYI> Utils::GenerateAlphaBasis()
{
	array<array<double, Ka>, TYI> result = array<array<double, Ka>, TYI>();
	for(short i = 0; i < 2*Ka; i+=2)
	{
		result[i / 2] = array<double, Ka>();
		for (short j = 0; j < TYI; ++j)
			result[i / 2][j] = RaisedCosine(j, i, Ka);
	}
	return result;
}

array<array<double, Kb>, TYO> Utils::GenerateBetaBasis()
{
	array<array<double, Kb>, TYO> result = array<array<double, Kb>, TYO>();
	for (short i = 0; i < 2 * Kb; i += 2)
	{
		result[i / 2] = array<double, Kb>();
		for (short j = 0; j < TYO; ++j)
			result[i / 2][j] = RaisedCosine(j, i, Kb);
	}
	return result;
}

/*
 Return a probability of spiking for every pixel
*/
array<float, NEURONS_IN> Utils::RateEncode(array<unsigned char, NEURONS_IN>& image)
{
	//auto imagePixels = GetPixelIntensities(imagePath);
	array<float, NEURONS_IN> probabilities = array<float, NEURONS_IN>();
	
	for (int i = 0; i < 784; ++i)
	{
		probabilities[i] = image[i] * (P_RANGE.second - P_RANGE.first) / 255 + P_RANGE.first;
	}
	return probabilities;
}

/*
 Return a array of size T that contains the train of spikes based on probability
*/
array<bool,T> Utils::GenerateSpikes(float probability)
{
	array<bool,T> train = array<bool,T>();

	//random seed

	probability = probability * 10000;

	for (int i = 0; i < T; ++i)
		if (rand() % 10000 <= probability)
			train[i] = 1;
		else
			train[i] = 0;

	return train;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/*
 Read the images/labels from the MNIST database
*/
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> ReadMNIST(int NumberOfImages, string imagesPath, string labelsPath)
{
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> arr = vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>(NumberOfImages);

	std::ifstream images_file(imagesPath, std::ios::binary);
	std::ifstream labels_file(labelsPath, std::ios::binary);

	if (images_file.is_open() && labels_file.is_open())
	{
		int magic_number_images = 0;
		int magic_number_labels = 0;
		int number_of_entries = 0;
		int n_rows = 0;
		int n_cols = 0;
		//read the magic numbers
		images_file.read((char*)&magic_number_images, sizeof(magic_number_images));
		magic_number_images = ReverseInt(magic_number_images);
		labels_file.read((char*)&magic_number_labels, sizeof(magic_number_labels));
		magic_number_labels = ReverseInt(magic_number_labels);
		//read the number of entries (in both files so the index is updated)
		images_file.read((char*)&number_of_entries, sizeof(number_of_entries));
		labels_file.read((char*)&number_of_entries, sizeof(number_of_entries));
		number_of_entries = ReverseInt(number_of_entries);
		//read rows and columns from image file
		images_file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		images_file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		//generate the array
		for (int i = 0; i < NumberOfImages; ++i)
		{
			arr[i].first = array<unsigned char, NEURONS_IN>();

			unsigned char tmp = 0;
			labels_file.read((char*)&tmp, sizeof(tmp));
			arr[i].second = tmp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
					arr[i].first[(n_rows*r) + c] = temp;
				}
			}
		}
		return arr;
	}
	else
		throw Exception();
}

/*
 Return the training data
*/
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::GetTrainingData(int NumberOfImages)
{
	auto data = ReadMNIST(NumberOfImages, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);

	//shuffle the array using Fisher–Yates algorithm
	int i = NumberOfImages - 1;
	int j;
	pair<array<unsigned char, NEURONS_IN>, unsigned char> temp;
	while (i > 0)
	{
		j = floor(((rand() % 10)/10.0) * (i + 1));
		temp = data[i];
		data[i] = data[j];
		data[j] = temp;
		i = i - 1;
	}

	return data;
}

/*
Return the test data
*/
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::GetTestData(int NumberOfImages)
{
	return ReadMNIST(NumberOfImages, TEST_IMAGES_PATH, TEST_LABELS_PATH);
}

void Utils::PrintLine(string&& str)
{
	lock.lock();
	const auto threadId = std::this_thread::get_id();
	std::cout << threadId << "> " << str << std::endl;
	lock.unlock();
}