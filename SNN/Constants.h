#pragma once
#include "stdafx.h"
#include <cmath>
#include <fstream>

using std::string;

// number of classes
const short CLASSES = 10;
// number of inputs
const short NEURONS_IN = 784;

// size of train
const short T = 4;
// length of relevant window
const short TYI = 3;
// number of alpha bases
const short Ka = TYI;
// length of relevant window
const short TYO = 2;
// number of beta bases
const short Kb = TYO;
// learning rate
const float LEARNING_RATE =  0.00005;
// range of probabilities for spike decoding
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