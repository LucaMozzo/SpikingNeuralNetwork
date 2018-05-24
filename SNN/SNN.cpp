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
	//only train on the first 5 digits
	array<unsigned char, 5> filter = { 0, 1, 2, 3, 4};
	n.Train<filter.size()>(1, 60000, nullptr);
	//export the training data
	n.ExportData("data.db");
	//validate against test dataset
	n.Validate<filter.size()>(10000, true);

	//getchar();
	return 0;
}