#pragma once
#include "stdafx.h"
#include <thread>
#include "Network.h"

using std::thread;
using std::vector;

class TrainingPool
{
protected:
	static vector<thread> runningInstances;

	static void InitTraining(short epochs, int trainingImages, string dbOut, bool validate);

public:

	static void TrainAsync(short epochs, int trainingImages, string dbOut, bool validate);
	static void Join();
};