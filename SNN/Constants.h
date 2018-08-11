#pragma once
#include "stdafx.h"
#include <cmath>
#include <fstream>

using std::string;
using std::array;

/*! \file This file contains the hyperparameters and constants used in the network*/

enum BasisFunctions
{
	BINARY, RAISED_COSINE
};

const short CLASSES = 10; /**< Number of classes*/
const short NEURONS_IN = 784; /**< Number of input neurons*/

const short T = 4; /**< Length of the spike train*/
const short TYI = 4; /**< Length of short-term memory window in presynaptic phase*/
const short Ka = TYI; /**< Number of alpha bases*/
const short TYO = 4; /**< Length of short-term memory window in feedback phase*/
const short Kb = TYO; /**< Number of beta bases*/
const float LEARNING_RATE =  0.001; /**< Learning rate*/
const std::pair<float, float> P_RANGE = { 0, 0.5 }; /**< Range of probabilities for spike decoding*/
const array<char, T> CORRECT_PATTERN = {0, 1, 0, 1}; /**< Pattern used to train the network (y)*/
const BasisFunctions BASIS_FUNCTION = BINARY;
const char PRECISION = 8; /**< Precision bits for quantization. 0 = disabled*/
const string TRAIN_IMAGES_PATH = "D:\\train-images.idx3-ubyte"; /**< Training images database path*/
const string TRAIN_LABELS_PATH = "D:\\train-labels.idx1-ubyte"; /**< Training labels database path*/
const string TEST_IMAGES_PATH = "D:\\t10k-images.idx3-ubyte"; /**< Test images database path*/
const string TEST_LABELS_PATH = "D:\\t10k-labels.idx1-ubyte"; /**< Test labels database path*/

/**
Sigmoid function used for activation
@param x Potential
@return Probability of spiking
*/
static const double g(const double x)
{
	return 1 / (1 + exp(-x));
}

/**
Approximated sigmoid function
@param x Potential
@return Probability of spiking
*/
static const double a_g(const double x)
{
	double decimal = abs(x - static_cast<int>(x));
	double val = (0.5 + decimal / 4.)/pow(2, abs(x));
	return x >= 0 ? val : 1 - val;
}