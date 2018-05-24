#pragma once
#include "stdafx.h"
#include <thread>
#include "Network.h"

using std::thread;
using std::vector;

/**
This calss contains methods for performing multiple training sessions on networks in parallel and exporting the results
*/
class TrainingPool
{
protected:
	static vector<thread> runningInstances; /**< A list of threads that are currently running */

	/**
	Method that makes the call to the Train function of the Network class
	@param epochs The number of epochs
	@param trainingImages The number of images to train the network on. The number should be 1-60000
	@param dbOut The path of the database where the data should be exported
	@param validate True if validation on the entire test set should be performed after the training
	*/
	static void InitTraining(short epochs, int trainingImages, string dbOut, bool validate);

public:

	/**
	Trains a network in a parallel thread and exports the training data
	@param epochs The number of epochs
	@param trainingImages The number of images to train the network on. The number should be 1-60000
	@param dbOut The path of the database where the data should be exported
	@param validate True if validation on the entire test set should be performed after the training
	*/
	static void TrainAsync(short epochs, int trainingImages, string dbOut, bool validate);
	/**
	Wait for all running threads to finish executing
	*/
	static void Join();
};