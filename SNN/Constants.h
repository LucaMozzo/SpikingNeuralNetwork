#pragma once
#include "stdafx.h"
#include <cmath>
#include <fstream>

using std::string;

/*! \file */

enum BasisFunctions
{
	BINARY, RAISED_COSINE
};

const short CLASSES = 10; /**< Number of classes*/
const short NEURONS_IN = 784; /**< Number of input neurons*/

const short T = 8; /**< Length of the spike train*/
const short TYI = 8; /**< Length of short-term memory window in presynaptic phase*/
const short Ka = TYI; /**< Number of alpha bases*/
const short TYO = 8; /**< Length of short-term memory window in feedback phase*/
const short Kb = TYO; /**< Number of beta bases*/
const float LEARNING_RATE = 0.001; /**< Learning rate*/
const std::pair<float, float> P_RANGE = { 0, 0.5 }; /**< Range of probabilities for spike decoding*/
const BasisFunctions BASIS_FUNCTION = BINARY;
const char PRECISION = 8; /**< Precision bits for quantization. 0 = disabled*/
const char LFSR_SEQ_LENGTH = 10; /**< Size of the LFSR seed sequence. 0 = use system random generator */
const char TAP_LENGTH = 2; /**< Length of the LFSR tap sequence */
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
	if (x < -8)
		return 0;
	if (x > 8)
		return 1;
	if (x <= 0) {
		return (0.5 + 0.25*(x + (fabs(ceil(x))))) / (pow(2, (fabs(ceil(x)))));
	}
	else if (x > 0)
		return 1 - ((0.5 + 0.25*(-x + (fabs(ceil(-x))))) / (pow(2, (fabs(ceil(-x))))));
}