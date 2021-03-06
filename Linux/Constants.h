#pragma once
#include <cmath>
#include <fstream>
#include <array>

using std::string;
using std::array;

/*! \file */

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
const string TRAIN_IMAGES_PATH = "train-images.idx3-ubyte"; /**< Training images database path*/
const string TRAIN_LABELS_PATH = "train-labels.idx1-ubyte"; /**< Training labels database path*/
const string TEST_IMAGES_PATH = "t10k-images.idx3-ubyte"; /**< Test images database path*/
const string TEST_LABELS_PATH = "t10k-labels.idx1-ubyte"; /**< Test labels database path*/

/**
Sigmoid function used for activation
@param x Potential
@return Probability of spiking
*/
static /*const*/ double g(const double x)
{
	return 1 / (1 + exp(-x));
}
