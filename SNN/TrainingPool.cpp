#include "stdafx.h"
#include "TrainingPool.h"
#include "Network.h"

vector<thread> TrainingPool::runningInstances(0);

void TrainingPool::InitTraining(short epochs, int trainingImages, string dbOut, bool validate)
{
	auto threadId = std::this_thread::get_id();
	Utils::PrintLine("Training started.");

	Network network{};
	network.Train<0>(epochs, trainingImages);
	network.ExportData(dbOut);

	Utils::PrintLine("Training completed.");
	/*auto itr = std::find_if(runningInstances.begin(), runningInstances.end(), [threadId](const thread& t) {
		if (t.get_id() == threadId)
			return true;
		return false;
	});

	if (itr != runningInstances.end())
		runningInstances.erase(itr);*/

	if(validate)
	{
		Utils::PrintLine("Validation started.");
		network.Validate<0>();
		Utils::PrintLine("Validation completed.");
	}
}

void TrainingPool::TrainAsync(short epochs, int trainingImages, string dbOut, bool validate)
{
	runningInstances.push_back(std::thread(InitTraining, epochs, trainingImages, dbOut, validate));
}

void TrainingPool::Join()
{
	for (auto& trd : runningInstances)
		trd.join();
}