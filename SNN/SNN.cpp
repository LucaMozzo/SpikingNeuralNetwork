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

	n.ImportData("4,3,2.db");
	for (int i = 0; i < 30; ++i) {
		n.Train(1);
		n.Validate();
		n.Validate(10000, false);
	}
	n.ExportData("4,3,2_2.db");

	getchar();
	return 0;
}