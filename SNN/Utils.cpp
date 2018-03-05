#include "stdafx.h"
#include "Utils.h"
#include "Constants.h"
#include <time.h>
#include <algorithm>
#include <opencv2\opencv.hpp>

using namespace cv;
using std::pair;

/*
 OBSOLETE
 Return the intensity of every pixel 0-255
*/
/*unsigned char* GetPixelIntensities(string imagePath)
{
	// Read the image file
	Mat image = imread(imagePath, IMREAD_GRAYSCALE);

	if (image.empty()) // Check for failure
	{
		throw new Exception();
	}

	unsigned char* pixels = new unsigned char[image.rows*image.cols];

	int index = 0;
	for(int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
			pixels[index++] = image.at<unsigned char>(i, j);

	return pixels;
}*/

/*
 Return a probability of spiking for every pixel
*/
float* Utils::RateEncode(unsigned char* image)
{
	//auto imagePixels = GetPixelIntensities(imagePath);
	float* probabilities = new float[784];
	
	for (int i = 0; i < 784; ++i)
	{
		probabilities[i] = image[i] * (P_RANGE.second - P_RANGE.first) / 255 + P_RANGE.first;
	}
	return probabilities;
}

/*
 Return a vector of size T that contains the train of spikes based on probability
*/
bool* Utils::GenerateSpikes(float probability)
{
	bool* train = new bool[T];

	//random seed
	srand(time(NULL));

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
pair<unsigned char*, unsigned char>* ReadMNIST(int NumberOfImages, string imagesPath, string labelsPath)
{
	pair<unsigned char*, unsigned char>* arr = new pair<unsigned char*, unsigned char>[NumberOfImages];

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
		for (int i = 0; i<NumberOfImages; ++i)
		{
			arr[i].first = new unsigned char[n_rows*n_cols];

			unsigned char temp = 0;
			labels_file.read((char*)&temp, sizeof(temp));
			arr[i].second = temp;
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					images_file.read((char*)&temp, sizeof(temp));
					arr[i].first[(n_rows*r) + c] = temp;
				}
			}
		}
		return arr;
	}
	return nullptr;
}


/*
 Return the training data
*/
pair<unsigned char*, unsigned char>* Utils::GetTrainingData(int NumberOfImages)
{
	auto data = ReadMNIST(NumberOfImages, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);

	srand(time(NULL));

	//shuffle the array using Fisher�Yates algorithm
	int i = NumberOfImages - 1;
	int j;
	pair<unsigned char*, unsigned char> temp;
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
pair<unsigned char*, unsigned char>* Utils::GetTestData(int NumberOfImages)
{
	return ReadMNIST(NumberOfImages, TEST_IMAGES_PATH, TEST_LABELS_PATH);
}