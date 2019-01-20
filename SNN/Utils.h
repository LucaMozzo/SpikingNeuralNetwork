#pragma once
#include "stdafx.h"
#include <mutex>

using std::array;
using std::vector;
using std::string;
using std::pair;

/**
This class contains general methods used in the programme
*/
class Utils 
{
private:

	static std::mutex lock; /**< Multithreading lock */
	/**
	Raised cosine function implementation
	@param time The x-value at which get the value
	@param mean The x-value where the value should be the highest
	@param stddev The standard deviation, i.e. distance from the mean where the numction is non-zero
	@return the value at time t of the curve
	*/
	static float RaisedCosine(int time, int mean, float stddev);
	static int ReverseInt(int i);
	/**
	Read from the MNIST database file
	@param FILTER_SIZE The number of elements in the filter
	@param NumberOfImages Number of images to read from the database
	@param imagesPath Path of the image database
	@param labelsPath Path of the labels database
	@param filter The filter to limit the labels in the training - if nullptr, all 10 digits are considered
	@param maxImagesPerLabel The maximum of images for each digit
	@return The a list of <image, label> pairs
	*/
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> ReadMNIST(int NumberOfImages, string imagesPath, string labelsPath, array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel);

public:

	/**
	Perform rate encode on the pixels of the image
	@param image The input image
	@return An array of probabilities of spiking
	*/
	static array<float, NEURONS_IN> RateEncode(array<unsigned char, NEURONS_IN>& image);
	/**
	Generate spikes trains from probabilities
	@param probability The probability of spiking
	@return The train of spikes
	*/
	static array<bool, T> GenerateSpikes(float probability);
	/**
	Returns the training data
	@param FILTER_SIZE The number of elements in the filter
	@param NumberOfImages Number of images to read from the database
	@param filter The filter to limit the labels in the training - if nullptr, all 10 digits are considered
	@param maxImagesPerLabel The maximum of images for each digit
	@return The a list of <image, label> pairs
	*/
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTrainingData(int NumberOfImages, array<unsigned char, FILTER_SIZE>* filter = nullptr, int maxImagesPerLabel = 0);
	/**
	Returns the test data
	@param FILTER_SIZE The number of elements in the filter
	@param NumberOfImages Number of images to read from the database
	@param filter The filter to limit the labels in the training - if nullptr, all 10 digits are considered
	@param maxImagesPerLabel The maximum of images for each digit
	@return The a list of <image, label> pairs
	*/
	template<std::size_t FILTER_SIZE>
	static vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> GetTestData(int NumberOfImages, array<unsigned char, FILTER_SIZE>* filter = nullptr, int maxImagesPerLabel = 0);
	/**
	Generates the basis matrix A
	@returns The basis matrix A
	*/
	static array<array<double, Ka>, TYI> GenerateAlphaBasis();
	/**
	Generates the basis matrix B
	@returns The basis matrix B
	*/
	static array<array<double, Kb>, TYO> GenerateBetaBasis();
	/**
	Thread-safe method for printing to terminal
	@param str The string to be printed to terminal
	*/
	static void PrintLine(string&& str);
	/**
Get the min and max weight in the given matrix
@param weights The matrix of weights
@returns The range of the weights
*/
	template<std::size_t ROWS, std::size_t COLS>
	static pair<double, double> GetMatrixRange(array<array<double, COLS>, ROWS>& weights);
	/**
	Get the min and max bias in the given array
	@param weights The matrix of biases
	@returns The range of the biases
	*/
	static pair<double, double> GetVectorRange(array<double, CLASSES>& biases);
	/**
	Get the step size for the weights quantization
	@param range The max and min value of the weights
	@returns The size of a step
	*/
	static double GetStepSize(pair<double, double>& range);
	/**
	Compute the quantized weights
	@param weights The matrix of weights
	@param stepSize The size of 1 step
	*/
	template<std::size_t ROWS, std::size_t COLS>
	static void QuantizeMatrix(array<array<double, COLS>, ROWS>& weights, double stepSize);
	/**
	Compute the quantized biases
	@param bias The vector of biases
	@param stepSize The size of 1 step
	*/
	static void QuantizeVector(array<double, CLASSES>& bias, double stepSize);
	/**
	LFSR for random number generator
	@param seed The initial LFSR values
	@param tap The value that affects the next bit position
	@returns The resulting sequence
	*/
	static vector<array<bool, LFSR_SEQ_LENGTH>> LFSR(array<bool, LFSR_SEQ_LENGTH> seed, const array<int, TAP_LENGTH> tap);
	/**
	Convert an array of boolean to the decimal representation
	@param binary The array of booleans
	@param offset The offset i.e. the positions to skip in the array
	@return The decimal representation
	*/
	template<std::size_t SIZE>
	static float BinaryToDec(array<bool, SIZE>& binary, char offset = 0);
};

template <std::size_t FILTER_SIZE>
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::ReadMNIST(int NumberOfImages, string imagesPath,
	string labelsPath, array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel)
{
	vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> arr = vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>>(0);

	int imagesPerLabel[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

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

			array<unsigned char, NEURONS_IN> imgdata{};
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
					imgdata[(n_rows*r) + c] = temp;
				}
			}

			if (filter != NULL && std::find(filter->begin(), filter->end(), tmp) == filter->end())
				continue; //skip image if not in the filter

			if (imagesPerLabel[tmp] >= maxImagesPerLabel && maxImagesPerLabel != 0)
				continue; //reached max number of images for that digit (0=no limit)
			//else add the pair to the vector
			pair<array<unsigned char, NEURONS_IN>, unsigned char> entry{};
			entry.first = imgdata;
			entry.second = tmp;
			arr.push_back(entry);
			++imagesPerLabel[tmp];
		}
		return arr;
	}
}


template <std::size_t FILTER_SIZE>
vector<pair<array<unsigned char, NEURONS_IN>, unsigned char>> Utils::GetTrainingData(int NumberOfImages,
	array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel)
{
	auto data = ReadMNIST<FILTER_SIZE>(NumberOfImages, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, filter, maxImagesPerLabel);

	//shuffle the array using Fisher–Yates algorithm
	int i = data.size() - 1;
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
	array<unsigned char, FILTER_SIZE>* filter, int maxImagesPerLabel)
{
	return ReadMNIST<FILTER_SIZE>(NumberOfImages, TEST_IMAGES_PATH, TEST_LABELS_PATH, filter, maxImagesPerLabel);
}

template<std::size_t ROWS, std::size_t COLS>
inline pair<double, double> Utils::GetMatrixRange(array<array<double, COLS>, ROWS>& weights)
{
	pair<double, double> range(INT_MAX, INT_MIN);
	for (short r = 0; r < ROWS; ++r)
		for (short c = 0; c < COLS; ++c)
			if (weights[r][c] < range.first)
				range.first = weights[r][c];
			else if (weights[r][c] > range.second)
				range.second = weights[r][c];
	return range;
}

template<std::size_t ROWS, std::size_t COLS>
inline void Utils::QuantizeMatrix(array<array<double, COLS>, ROWS>& weights, double stepSize)
{
	for (short r = 0; r < ROWS; ++r)
		for (short c = 0; c < COLS; ++c)
			weights[r][c] = stepSize * round(weights[r][c] / stepSize);
}

template<std::size_t SIZE>
inline float Utils::BinaryToDec(array<bool, SIZE>& binary, char offset)
{
	char arrayIndex = offset;
	float dec = 0;
	for (int i = 1; i < SIZE - offset; ++i)
		dec = dec + binary[arrayIndex++] * pow(2, -1 * (i));

	return dec;
}