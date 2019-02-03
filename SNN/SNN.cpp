#include "stdafx.h"
#include "Network.h"
#include "Utils.h"
#include <time.h>	
#include <opencv2/opencv.hpp>
#include "TrainingPool.h"

using namespace cv;
using namespace std;
/*! \file */

/**
This is where your program should be written
*/
int main(int argc, char** argv)
{
	srand(time(NULL));

	string command = "echo \"\" > " + BASE_NAME + ".csv";
	system(command.c_str());

	//create a network
	auto n = Network();
	n.Train<0>(5, 20000);
	n.Validate<0>();
	n.Train<0>(5, 20000);
	n.Validate<0>();

	//only train on the first 5 digits
	//n.TrainVal(200, 1000, 1000, true);
	n.ExportFile();
	Utils::PrintLine("Done");
	getchar();
	return 0;
}