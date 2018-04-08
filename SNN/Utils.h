#pragma once
#include "stdafx.h"
#include <mutex>

using std::array;
using std::vector;
using std::string;
using std::pair;

class Utils 
{
private:

	static std::mutex lock;
	static float RaisedCosine(int time, int mean, float stddev);
	static int ReverseInt(int i);
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> ReadMNIST(int NumberOfImages, string imagesPath, string labelsPath, array<unsigned char, FILTER_SIZE>* filter);

public:

	static array<float, NEURONS_IN> RateEncode(array<unsigned char, NEURONS_IN>& image);
	static array<bool, T> GenerateSpikes(float probability);
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTrainingData(int NumberOfImages, array<unsigned char, FILTER_SIZE>* filter=nullptr);
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTestData(int NumberOfImages, array<unsigned char, FILTER_SIZE>* filter=nullptr);
	static array<float, T> GenerateBasisMatrix(short meanOffset);
	static array<array<double, Ka>, TYI> GenerateAlphaBasis();
	static array<array<double, Kb>, TYO> GenerateBetaBasis();
	static void PrintLine(string&& str);
};

template <std::size_t FILTER_SIZE>
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::ReadMNIST(int NumberOfImages, string imagesPath,
	string labelsPath, array<unsigned char, FILTER_SIZE>* filter)
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
			unsigned char tmp = 0;
			labels_file.read((char*)&tmp, sizeof(tmp));
			arr[i].second = tmp;

			if (filter != NULL && std::find(filter->begin(), filter->end(), tmp) == filter->end())
				continue; //skip image if not in the filter

			arr[i].first = array<unsigned char, NEURONS_IN>();
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

template <std::size_t FILTER_SIZE>
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::GetTrainingData(int NumberOfImages,
	array<unsigned char, FILTER_SIZE>* filter)
{
	auto data = ReadMNIST<FILTER_SIZE>(NumberOfImages, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, filter);

	//shuffle the array using Fisher–Yates algorithm
	int i = NumberOfImages - 1;
	int j;
	pair<array<unsigned char, NEURONS_IN>, unsigned char> temp;
	while (i > 0)
	{
		j = floor(((rand() % 10) / 10.0) * (i + 1));
		temp = data[i];
		data[i] = data[j];
		data[j] = temp;
		i = i - 1;
	}

	return data;
}

template <std::size_t FILTER_SIZE>
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::GetTestData(int NumberOfImages,
	array<unsigned char, FILTER_SIZE>* filter)
{
	return ReadMNIST<FILTER_SIZE>(NumberOfImages, TEST_IMAGES_PATH, TEST_LABELS_PATH, filter);
}

