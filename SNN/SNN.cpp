#include "stdafx.h"
#include "Network.h"
#include "Utils.h"
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include<time.h>	
#include "TrainingPool.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	srand(time(NULL));

	LARGE_INTEGER beginTime;
	QueryPerformanceCounter(&beginTime);
	auto n = Network();
	/*TrainingPool::TrainAsync(2, 60000, "db7.db", true);
	TrainingPool::TrainAsync(1, 10000, "db8.db", true);
	n.Train(1, 60000);

	n.ExportData("data1.db");

	Utils::PrintLine("Validating mainthread result");
	n.Validate(10000);
	TrainingPool::Join();*/
	n.CrossValidate();
	
	/*n.ImportData("18,8,4.db");
	n.Train(30, 60000);
	n.ExportData("18,8,4_2.db");
	n.Validate();*/

	LARGE_INTEGER endTime;
	QueryPerformanceCounter(&endTime);
	LARGE_INTEGER timerFreq;
	QueryPerformanceFrequency(&timerFreq);
	double freq = 1.0f / timerFreq.QuadPart;

	double timeSeconds = (endTime.QuadPart - beginTime.QuadPart) * freq;
	cout << "\n\nMainThread Ended: " << timeSeconds << endl;
	getchar();
	return 0;
}