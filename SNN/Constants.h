#pragma once
#include "stdafx.h"
#include <cmath>
#include <fstream>

using std::string;
using std::vector;

const short CLASSES = 10;
const short NEURONS_IN = 784;

const short T = 6;
const short TYI = 4;
const short TYO = 2;
const float LEARNING_RATE = 0.0001;
const std::pair<float, float> P_RANGE = { 0, 0.5 };

// Resource paths
const string TRAIN_IMAGES_PATH = "D:\\train-images.idx3-ubyte";
const string TRAIN_LABELS_PATH = "D:\\train-labels.idx1-ubyte";
const string TEST_IMAGES_PATH = "D:\\t10k-images.idx3-ubyte";
const string TEST_LABELS_PATH = "D:\\t10k-labels.idx1-ubyte";

// Sigmoid function
static const double g(const double x)
{
	return 1 / (1 + exp(-x));
}