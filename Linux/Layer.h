#pragma once
#include "MatrixOps.h"

using std::array;
/**
This class represents an input layer in the neural network and contains all the methods to perform operations on it
*/
class InputLayer
{
protected:

	array<array<bool, T>, NEURONS_IN> trains; /**< The trains of spikes generated by rate encoding process */
	array<array<double, Ka>, TYI> basis; /**< The basis matrix A contains the basis functions to apply to the weights w */
	short index = 0; /**< Index of next train in the array */

public:

	array<array<double, TYI>, CLASSES*NEURONS_IN> w; /**< The weights w for every synapse*/

	/**
	The cosntructor initialises the values of the weights and the basis matrix
	*/
	InputLayer();
	/**
	Adds a train of spikes to the array, as position [index]
	@param train The train of spikes
	*/
	void AddTrain(array<bool, T>& train);
	/**
	Reset the values of the trains and index
	*/
	void ResetTrains();
	/**
	Convolve alphas with the trains and apply the basis matrix.
	This is one part of the process to compute the potential of the output neuron
	@return The preprocessed spike trains
	*/
	array<array<double, T-1>, CLASSES*NEURONS_IN> ApplyAlphas();
	/**
	Updates the weights w using the errors
	@params errors The errors for updating the weights
	*/
	void UpdateAlphas(array<array<double, T>, CLASSES>& errors);
};

/**
This class represents an output layer in the neural network and contains all the methods to perform operations on it
*/
class OutputLayer
{
protected:

	array<array<double, T>, CLASSES> u; /**< The potential of every output neuron at every time t */
	array<array<bool, T>, CLASSES> y; /**< The output spike trains of every class */
	array<array<double, Kb>, TYO> basis; /**< The basis matrix B contains the basis functions to apply to the weights v */

public:

	array<array<double, TYO>, CLASSES> v; /**< The weights of the feedback kernel*/
	array<double, CLASSES> gammas; /**< The biases of every class*/

	/**
	The cosntructor initialises the values of the weights and the basis matrix
	*/
	OutputLayer();
	/**
	Reset the values of the potential and output
	*/
	void Reset();
	/**
	Computes the potential and output starting from the preprocessed trains
	@param synapsesOut The preprocessed spike trains from the input layer
	*/
	void ComputeOutput(array<array<double, T-1>, CLASSES*NEURONS_IN>& synapsesOut, signed char label = -1);
	/**
	Computes the errors at every time t for the given class
	@params label The class for which compute the errors
	@returns An array of errors for every class label
	*/
	array<array<double, T>, CLASSES> ComputeErrors(unsigned char label) const;
	/**
	Returns the "Winner" of the network: uses rate decoding to determine the class with the most spikes
	@return The prediction of the network
	*/
	char ComputeWinner() const;
	/**
	Updates the weights v using the errors
	@params errors The errors for updating the weights
	@see ComputeErrors
	*/
	void UpdateBetas(array<array<double, T>, CLASSES>& errors);
	/**
	Updates the biases using the errors
	@params errors The errors for updating the biases
	@see ComputeErrors
	*/
	void UpdateGammas(array<array<double, T>, CLASSES>& errors);
};
