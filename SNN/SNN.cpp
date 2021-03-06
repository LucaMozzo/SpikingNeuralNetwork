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

	//create a network
	auto n = Network();

	n.TrainVal(100, 1000, 1000, true);

	Utils::PrintLine("Done");
	getchar();
	return 0;
}