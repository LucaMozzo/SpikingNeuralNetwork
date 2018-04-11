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

	array<unsigned char, 0> filter = { };
	/*n.ExportData("a.db");
	n.ImportData("a.db");*/
	n.Train<filter.size()>(1, 500, nullptr/*, &filter*/);
	n.Validate<filter.size()>(10000, true);
	n.ExportData("a.db");
	n = Network();
	n.ImportData("a.db");
	n.Validate<filter.size()>(10000, true);
	/*n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);
	n.Validate<filter.size()>(10000, true);*/

	getchar();
	return 0;
}