#include "stdafx.h"
#include "Network.h"
#include "Utils.h"
#include <time.h>	
#include <opencv2/opencv.hpp>
#include "TrainingPool.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	srand(time(NULL));

	//create a network
	auto n = Network();

	array<unsigned char, 1> filter = { 1};
	n.Train<filter.size()>(1, 60000, nullptr/*, &filter*/);
	//n.Validate<filter.size()>(10000, true/*, &filter*/);
	//n.Validate<filter.size()>(60000, false/*, &filter*/);

	//getchar();
	return 0;
}